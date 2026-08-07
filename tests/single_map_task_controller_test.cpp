#include "task/single_map_task_controller.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

SingleMapModelIdentity Model(int version, int64_t trained_samples) {
    SingleMapModelIdentity model;
    model.model_version = version;
    model.model_checksum = std::string(64, static_cast<char>('a' + version % 20));
    model.trained_samples = trained_samples;
    return model;
}

std::vector<SingleMapAgentEvaluation> Agents(int successes,
                                             int success_steps,
                                             int failure_steps) {
    std::vector<SingleMapAgentEvaluation> agents(4);
    for (int i = 0; i < 4; ++i) {
        agents[i].success = i < successes;
        agents[i].transition_count =
            agents[i].success ? success_steps : failure_steps;
    }
    return agents;
}

void CompleteRound(SingleMapTaskController& controller,
                   maze::EpisodeMode mode,
                   const SingleMapModelIdentity& model,
                   int successes_per_episode,
                   int success_steps,
                   int failure_steps) {
    for (int episode = 0; episode < 25; ++episode) {
        std::string error;
        assert(controller.RecordEvaluationEpisode(
            mode, model,
            Agents(successes_per_episode, success_steps, failure_steps),
            error));
    }
}

void CompletePassingCampaign(SingleMapTaskController& controller,
                             const SingleMapModelIdentity& model,
                             int success_steps,
                             int max_steps) {
    CompleteRound(controller, maze::EPISODE_MODE_EVALUATION_ARGMAX,
                  model, 4, success_steps, max_steps);
    assert(controller.GetSnapshot().evaluation_round == 2);
    CompleteRound(controller, maze::EPISODE_MODE_EVALUATION_ARGMAX,
                  model, 4, success_steps, max_steps);
    assert(controller.GetSnapshot().evaluation_round == 3);
    // The stochastic diagnostic does not participate in the Gate.
    CompleteRound(controller, maze::EPISODE_MODE_EVALUATION_STOCHASTIC,
                  model, 0, success_steps, max_steps);
}

}  // namespace

int main() {
    std::string error;
    SingleMapTaskController controller;
    assert(controller.Initialize(188, Model(0, 0), 0, error));

    SingleMapEpisodePlan plan;
    assert(controller.PlanNextEpisode(Model(0, 0), 0, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_8X);
    assert(plan.max_steps == 1504);

    bool should_pause = false;
    assert(controller.ShouldPauseTrainingCollection(
        Model(195, 99840), 99840, should_pause, error));
    assert(!should_pause);
    assert(controller.ShouldPauseTrainingCollection(
        Model(196, 100352), 100352, should_pause, error));
    assert(should_pause);

    const auto first_eval_model = Model(196, 100352);
    assert(controller.PlanNextEpisode(
        first_eval_model, 100352, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_EVALUATION_ARGMAX);
    assert(plan.model.model_version == 196);
    assert(controller.IsEvaluationActive());
    assert(controller.GetSnapshot().evaluation_round == 1);

    // A newer publication cannot replace the model pinned by the campaign.
    const auto republished_model = Model(197, 100352);
    assert(controller.PlanNextEpisode(republished_model, 100352,
                                      plan, error));
    assert(plan.model.model_version == 196);
    CompletePassingCampaign(controller, first_eval_model, 250, 1504);

    assert(controller.PlanNextEpisode(republished_model, 100352,
                                      plan, error));
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_4X);
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    assert(plan.max_steps == 752);

    const auto second_eval_model = Model(391, 200192);
    assert(controller.PlanNextEpisode(
        second_eval_model, 200192, plan, error));
    CompletePassingCampaign(controller, second_eval_model, 240, 752);
    assert(controller.PlanNextEpisode(second_eval_model, 200192,
                                      plan, error));
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_2X);
    assert(plan.max_steps == 376);

    const auto final_eval_model = Model(586, 300032);
    assert(controller.PlanNextEpisode(
        final_eval_model, 300032, plan, error));
    CompletePassingCampaign(controller, final_eval_model, 280, 376);
    assert(controller.PlanNextEpisode(final_eval_model, 300032,
                                      plan, error));
    assert(!plan.continue_task);
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_COMPLETE);
    assert(controller.GetSnapshot().complete);
    assert(controller.history().size() == 3);

    // A failed final attempt closes the current seed as a quality failure.
    SingleMapTaskControllerConfig short_config;
    short_config.stage_8x_sample_budget = 1000;
    short_config.stage_4x_sample_budget = 1000;
    short_config.stage_2x_sample_budget = 1000;
    short_config.evaluation_interval_samples = 100000;
    SingleMapTaskController failed(short_config);
    error.clear();
    assert(failed.Initialize(188, Model(0, 0), 0, error));
    assert(failed.PlanNextEpisode(Model(0, 0), 0, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    assert(failed.ShouldPauseTrainingCollection(
        Model(1, 512), 512, should_pause, error));
    assert(should_pause);
    assert(failed.PlanNextEpisode(Model(1, 512), 512, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_EVALUATION_ARGMAX);
    CompleteRound(failed, maze::EPISODE_MODE_EVALUATION_ARGMAX,
                  Model(1, 512), 0, 188, 1504);
    CompleteRound(failed, maze::EPISODE_MODE_EVALUATION_ARGMAX,
                  Model(1, 512), 0, 188, 1504);
    CompleteRound(failed, maze::EPISODE_MODE_EVALUATION_STOCHASTIC,
                  Model(1, 512), 4, 188, 1504);
    assert(failed.PlanNextEpisode(Model(1, 512), 512, plan, error));
    assert(!plan.continue_task);
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_FAILED);
    assert(failed.GetSnapshot().failed);

    // The regression that left 256 samples untrained at the final campaign:
    // evaluation remains blocked until a homogeneous PPO batch is drained.
    SingleMapTaskController near_cap;
    error.clear();
    assert(near_cap.Initialize(188, Model(0, 0), 0, error));
    assert(near_cap.PlanNextEpisode(
        Model(1950, 998400), 998656, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    assert(near_cap.ShouldPauseTrainingCollection(
        Model(1950, 998400), 998656, should_pause, error));
    assert(!should_pause);
    assert(near_cap.ShouldPauseTrainingCollection(
        Model(1953, 999936), 999936, should_pause, error));
    assert(should_pause);
    assert(near_cap.PlanNextEpisode(
        Model(1953, 999936), 999936, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_EVALUATION_ARGMAX);

    // Whole-fragment ingress and a bounded variable learner batch must stop at
    // the last safe trainable window instead of crossing the hard stage cap.
    SingleMapTaskControllerConfig bounded_config;
    bounded_config.evaluation_interval_samples = 2000000;
    SingleMapTaskController bounded_cap(bounded_config);
    error.clear();
    assert(bounded_cap.Initialize(188, Model(0, 0), 0, error));
    assert(bounded_cap.ShouldPauseTrainingCollection(
        Model(1950, 998400), 998900, should_pause, error));
    assert(!should_pause);
    assert(bounded_cap.ShouldPauseTrainingCollection(
        Model(1950, 998400), 999028, should_pause, error));
    assert(should_pause);
    assert(bounded_cap.PlanNextEpisode(
        Model(1951, 999028), 999028, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_EVALUATION_ARGMAX);
    assert(bounded_cap.GetSnapshot().stage_produced_samples == 999028);

    // Sample Pool freshness is resolved after production. Reconcile that
    // explicit disposition inside the current stage without weakening the
    // monotonic counter check used by later updates.
    SingleMapTaskController freshness;
    error.clear();
    assert(freshness.Initialize(188, Model(0, 0), 0, error));
    assert(freshness.ShouldPauseTrainingCollection(
        Model(1, 512), 640, should_pause, error));
    assert(!should_pause);
    assert(freshness.ReconcileDiscardedTrainingSamples(128, error));
    assert(freshness.GetSnapshot().stage_produced_samples == 512);
    assert(freshness.ShouldPauseTrainingCollection(
        Model(1, 512), 512, should_pause, error));
    assert(!should_pause);
    assert(!freshness.ReconcileDiscardedTrainingSamples(513, error));

    // Once evaluation is due, collection alternates between pause-and-drain
    // and filling a stranded sub-batch remainder. This reaches an exact empty
    // ledger without admitting partial PPO updates.
    SingleMapTaskController periodic;
    error.clear();
    assert(periodic.Initialize(188, Model(0, 0), 0, error));
    assert(periodic.ShouldPauseTrainingCollection(
        Model(196, 100352), 100608, should_pause, error));
    assert(!should_pause);
    assert(periodic.ShouldPauseTrainingCollection(
        Model(196, 100352), 100864, should_pause, error));
    assert(should_pause);
    assert(periodic.ReconcileDiscardedTrainingSamples(128, error));
    assert(periodic.ShouldPauseTrainingCollection(
        Model(196, 100352), 100736, should_pause, error));
    assert(!should_pause);
    assert(periodic.ShouldPauseTrainingCollection(
        Model(196, 100352), 100864, should_pause, error));
    assert(should_pause);
    assert(periodic.ShouldPauseTrainingCollection(
        Model(197, 100864), 100864, should_pause, error));
    assert(should_pause);

    std::cout << "single-map task controller contract passed\n";
    return 0;
}
