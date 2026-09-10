#pragma once

#include "task/protocol/training_task_service.h"
#include "maze/protocol/adapter.h"
#include "proto/maze/maze.sdk.pb.h"

using MazeTaskService = TrainingTaskService<rl::task::maze::v1::MazeTaskServiceProtocol, MazeTaskAdapter>;
