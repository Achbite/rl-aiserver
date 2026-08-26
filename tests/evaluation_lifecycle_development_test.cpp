#include "grpc/maze_service.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

struct MazeServiceLifecycleTestAccess {
    static void MarkEvaluationInferenceReady(MazeServiceImpl& service) {
        service.state_.store(training::AISERVER_STATE_READY);
        service.model_state_.store(training::MODEL_STATE_READY);
        service.model_manifest_ = ModelManifest{};
        service.model_manifest_.sha256 = std::string(64, 'a');
    }

    static SessionManager::Session* CreateSession(MazeServiceImpl& service) {
        const std::string session_id = service.session_mgr_.CreateSession();
        return service.session_mgr_.GetSession(session_id);
    }

    static bool TaskStopRequested(const MazeServiceImpl& service) {
        return service.task_stop_requested_;
    }
};

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

AIServerConfig MakeEvaluationConfig() {
    AIServerConfig config;
    config.server.run_mode = aiserver_mode::kEvaluation;
    config.environment.agent_count = 1;
    config.task.episode_max_steps = 1504;
    config.policy.policy_spec_digest.hex = std::string(64, 'b');
    return config;
}

void FillCommand(const SessionManager::Session& session,
                 std::uint64_t sequence,
                 const std::string& idempotency_key,
                 const std::string& episode_id,
                 maze::LifecycleCommand* command) {
    command->mutable_task()->CopyFrom(session.task);
    command->set_session_id(session.session_id);
    command->set_episode_id(episode_id);
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(sequence);
    command->set_idempotency_key(idempotency_key);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
}

void TestEvaluationContinuesAfterCompletedEpisode() {
    MazeServiceImpl service(MakeEvaluationConfig());
    MazeServiceLifecycleTestAccess::MarkEvaluationInferenceReady(service);

    auto* session = MazeServiceLifecycleTestAccess::CreateSession(service);
    Require(session != nullptr, "create the evaluation Session");
    session->lifecycle_epoch = 1;
    session->initialized = true;
    session->task_state = maze::TASK_STATE_EVALUATING;
    session->session_state = maze::SESSION_STATE_EPISODE_ACTIVE;
    session->protocol_episode_state =
        maze::EPISODE_STATE_TERMINAL_REPORTED;
    session->episode_state = SessionManager::EpisodeState::Active;
    session->current_episode_id = "evaluation-episode-1";
    session->current_episode_mode = maze::EPISODE_MODE_EVALUATION;
    session->shortest_action_steps = 10;

    auto& agent = session->agents[0];
    agent.done_collected = true;
    agent.final_termination_reason =
        maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
    agent.episode_transition_count = 1504;
    agent.visited.insert(0);
    agent.terminal_frame_id = 1504;
    agent.observation_grid_x = 44;
    agent.observation_grid_y = 25;

    maze::EndEpisodeReq end_request;
    FillCommand(*session, 1, "end-evaluation-episode-1",
                session->current_episode_id,
                end_request.mutable_command());
    maze::EndEpisodeRsp end_response;
    service.EndEpisode(nullptr, &end_request, &end_response);

    Require(end_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED,
            "commit the completed evaluation Episode");
    Require(end_response.outcome().episode_id() ==
                "evaluation-episode-1" &&
                end_response.outcome().agents_size() == 1 &&
                end_response.outcome().agents(0).termination_reason() ==
                    maze::MAZE_TERMINATION_REASON_TIME_LIMIT,
            "preserve the completed evaluation outcome");
    Require(!MazeServiceLifecycleTestAccess::TaskStopRequested(service),
            "a completed evaluation Episode does not stop the workload");
    Require(session->task_state == maze::TASK_STATE_EVALUATING &&
                session->session_state == maze::SESSION_STATE_IDLE &&
                session->protocol_episode_state ==
                    maze::EPISODE_STATE_COMMITTED,
            "return the evaluation Session to the idle committed state");

    maze::BeginEpisodeReq begin_request;
    FillCommand(*session, 2, "begin-evaluation-episode-2", "",
                begin_request.mutable_command());
    maze::BeginEpisodeRsp begin_response;
    service.BeginEpisode(nullptr, &begin_request, &begin_response);

    Require(begin_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED,
            "begin the next evaluation Episode");
    Require(begin_response.assignment().continue_task() &&
                begin_response.assignment().mode() ==
                    maze::EPISODE_MODE_EVALUATION &&
                !begin_response.assignment().episode_id().empty() &&
                begin_response.assignment().episode_id() !=
                    "evaluation-episode-1",
            "assign a new evaluation Episode instead of completing the task");
    Require(session->task_state == maze::TASK_STATE_EVALUATING &&
                session->session_state ==
                    maze::SESSION_STATE_EPISODE_ACTIVE &&
                session->protocol_episode_state ==
                    maze::EPISODE_STATE_RUNNING,
            "advance the Session into the next evaluation Episode");
}

}  // namespace

int main() {
    TestEvaluationContinuesAfterCompletedEpisode();
    std::cout << "aiserver_evaluation_lifecycle_development_contract: PASS"
              << std::endl;
    return 0;
}
