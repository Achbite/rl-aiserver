#pragma once

#include "task/reward/reward_result.h"
#include "proto/communication/session.pb.h"
#include <string>
#include <vector>

// AIServer-internal inputs. They are produced by the task adapter, never Client.
struct AgentTaskInput {
    int agent_id = 0;
    bool terminal = false;
    std::vector<float> observation;
    std::vector<bool> action_mask;
    RewardResult reward;
};

struct ModelTaskAction {
    int agent_id;
    int action;
};

struct TaskError {
    enum class Kind { Input, Runtime, Rollout };
    rl::session::v1::CommandErrorCode code = rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT;
    std::string message;
    Kind kind = Kind::Input;
    bool Set(rl::session::v1::CommandErrorCode value, std::string detail,
             Kind failure = Kind::Input) {
        code = value;
        message = std::move(detail);
        kind = failure;
        return false;
    }
};
