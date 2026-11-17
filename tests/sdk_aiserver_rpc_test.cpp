#include "maze/protocol/service.h"
#include "maze/config/config_loader.h"
#include "maze/action/action_receipt.h"
#include "rl_sdk/task_client.h"
#include "proto/maze/maze.sdk.pb.h"
#include "model_distributor_fixture.h"
#include <cstdlib>
#include <iostream>

namespace {
void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}
using Client = rl_sdk::TaskClient<maze::MazeTaskServiceProtocol>;

void TestActions(const std::string& model, bool synchronous = false) {
    model_fixture::TemporaryRoot root;
    const auto bytes = model_fixture::ReadFile(model);
    model_fixture::FixedModelDistributor models(model_fixture::MakeManifest(bytes), bytes);
    model_fixture::CapturingSamplePool pool;
    if (synchronous) pool.SetSynchronous();
    model_fixture::LocalServer dependencies({&models, &pool});
    const auto map_path = root.path() / "routing.json";
    std::ofstream(map_path) << R"({"map_id":"routing","grid_size":1,"grid_cols":5,"grid_rows":5,
        "start_grid":{"x":1,"y":1},"goal_grid":{"x":4,"y":4},
        "blocked_bitmap_encoding":"row-major-u8-0-open-1-blocked","blocked_bitmap_hex":"00000000000000000000000000000000000000000000000000","bounds":{"x_max":5,"y_max":5},"start_pos":{"x":1.5,"y":1.5},"end_pos":{"x":4.5,"y":4.5},"walls":[{"x1":0,"y1":0,"x2":5,"y2":0,"thickness":0.1}]})";
    ClientConfig client_config;
    client_config.run.agent_num = 2;
    client_config.env.map_file = map_path.string();
    client_config.env.max_steps = 2;
    MazeEnv environment;
    Require(environment.Init(client_config), "load the real Client environment");
    // Seed a second distinguishable, valid environment position through Step.
    std::string error;
    Require(environment.Step(1, maze::MAZE_ACTION_RIGHT, error), error);

    MazeConfig config;
    config.server.run_mode = synchronous ? aiserver_mode::kTraining : aiserver_mode::kEvaluation;
    config.model.local_train_dir = (root.path() / "train").string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = dependencies.port;
    config.sample_distributor.host = "127.0.0.1";
    config.sample_distributor.port = dependencies.port;
    config.rollout.collection_batch_size = synchronous ? 1 : 0;
    config.model.evaluation_model_path = model;
    config.model.expected_obs_dim = 17;
    config.model.expected_action_dim = 9;
    config.environment.agent_count = 2;
    config.task.fixed_map_id = "routing";
    config.task.episode_max_steps = 2;
    config.policy.action_mask_mode = "required";
    config.policy.training_temperature = 1;
    config.observation.ray_max_range = 4;
    MazeTaskService service(config);
    Require(service.Start(), "start actual evaluation AIServer");
    model_fixture::LocalServer server({static_cast<maze::MazeTaskService::Service*>(&service)});
    Client client(model_fixture::ClientOptions());
    Require(client.Connect("127.0.0.1:" + std::to_string(server.port)), "connect actual SDK");
    maze::OpenSessionRsp open;
    Require(client.OpenSession(open) == rl_sdk::CommandOutcome::Applied, client.error());
    Require(open.environment().agent_count() == 2, "actual Open assignment gives both agent identities");
    maze::InitReq init;
    auto* map = init.mutable_map();
    map->set_map_id(environment.GetMapId());
    map->set_grid_columns(environment.GetGridCols());
    map->set_grid_rows(environment.GetGridRows());
    map->set_grid_size_microunits(environment.GetGridSizeMicrounits());
    map->set_start_grid_x(environment.GetStartGridX());
    map->set_start_grid_y(environment.GetStartGridY());
    map->set_goal_grid_x(environment.GetGoalGridX());
    map->set_goal_grid_y(environment.GetGoalGridY());
    map->set_blocked_bitmap(environment.GetBlockedBitmap());
    maze::InitRsp initialized;
    Require(client.Init(init, initialized) == rl_sdk::CommandOutcome::Applied, client.error());
    maze::BeginEpisodeRsp begin;
    Require(client.BeginEpisode(begin) == rl_sdk::CommandOutcome::Applied, client.error());
    Require(begin.assignment().max_steps() == 2,
            "server assignment reaches the Client");
    environment.SetMaxSteps(begin.assignment().max_steps());
    std::vector<maze_client::AgentExecutionCursor> cursors(2);
    for (int frame = 0; frame <= 2; ++frame) {
        maze::UpdateReq request;
        request.set_frame_id(environment.GetFrameId());
        for (int id : {1, 0}) {  // Response order must never stand in for agent_id.
            const auto& agent = environment.GetAgent(id);
            auto* state = request.add_agents();
            state->set_agent_id(id);
            state->mutable_position()->set_x(agent.grid_x);
            state->mutable_position()->set_y(agent.grid_y);
            state->set_is_done(agent.done);
            state->set_last_move_blocked(agent.last_move_blocked);
            state->set_termination_reason(agent.done ? maze::MAZE_TERMINATION_REASON_TIME_LIMIT
                                                     : maze::MAZE_TERMINATION_REASON_ACTIVE);
            if (!agent.done) for (bool available : environment.GetActionMask(id)) state->add_action_mask(available);
            Require(maze_client::AttachExecutedActionReceipt(frame, cursors[id], state), "attach actual action receipt");
        }
        maze::UpdateRsp response;
        std::vector<maze_client::AgentExecutionCursor> candidate;
        auto update = std::async(std::launch::async, [&] {
            return client.Update(request, response, [&](const auto& req, const auto& rsp) {
                    return maze_client::PrepareAppliedAgentUpdate(req, rsp, cursors, candidate);
                });
        });
        if (synchronous && frame > 0) {
            Require(pool.WaitForEnvelopeCount(frame * 2), "whole frame closes both actual Actor segments");
            const auto envelopes = pool.envelopes();
            Require(envelopes.back().collection_round_batch_size() == 1 &&
                    envelopes.back().collection_round_transition_count() == 2,
                    "last envelope seals the full frame including its explicit excess item");
            if (frame == 1) models.SetCandidate(frame, bytes);
            else pool.CompleteRound(envelopes.back().behavior_model());
            Require(update.wait_for(std::chrono::milliseconds(350)) == std::future_status::timeout &&
                    pool.envelopes().size() == static_cast<size_t>(frame * 2),
                    "WAIT retries neither execute actions nor enqueue the frame twice; both facts are required");
            if (frame == 1) pool.CompleteRound(envelopes.back().behavior_model());
            else models.SetCandidate(frame, bytes);
        }
        Require(update.get() == rl_sdk::CommandOutcome::Applied,
                "actual AIServer accepts state and receipt: " + client.error());
        cursors = std::move(candidate);
        if (frame == 2) {
            Require(environment.AllDone() && response.action_batch().actions_size() == 0 &&
                    response.reply().phase() == rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL,
                    "last Update acknowledges both terminal agents without extra actions");
            break;
        }
        Require(response.action_batch().actions_size() == 2, "both agents receive actions");
        for (const auto& action : response.action_batch().actions()) {
            const int id = action.agent_id();
            // x=1/4: UP logit=1, RIGHT=.5; x>=2/4: RIGHT wins.
            Require(synchronous || action.action_id() == (id == 0 ? maze::MAZE_ACTION_UP : maze::MAZE_ACTION_RIGHT),
                    "known ONNX argmax is routed to its original agent");
            Require(maze_client::ExecuteAssignedAction(environment, cursors[id], action, error), error);
        }
        environment.AdvanceFrame();
        Require(synchronous || (environment.GetAgent(0).grid_x == 1 && environment.GetAgent(0).grid_y == 2 + frame &&
                environment.GetAgent(1).grid_x == 3 + frame && environment.GetAgent(1).grid_y == 1),
                "real environment executes the respective model actions");
    }
    maze::EndEpisodeRsp ended;
    maze::CloseSessionRsp closed;
    Require(client.EndEpisode(ended) == rl_sdk::CommandOutcome::Applied &&
            client.CloseSession(closed) == rl_sdk::CommandOutcome::Applied, client.error());
    Require(synchronous ? model_fixture::FinishMetrics([&] { return service.BeginShutdown(); },
                service.Metrics(), service.MetricSourceIdentity()) : service.BeginShutdown(),
            "service shuts down cleanly");
}
}

int main(int argc, char** argv) {
    Require(argc == 2, "usage: sdk_aiserver_rpc_test MODEL");
    TestActions(argv[1]);
    TestActions(argv[1], true);
    std::cout << "aiserver_sdk_action_data_path: PASS\n";
}
