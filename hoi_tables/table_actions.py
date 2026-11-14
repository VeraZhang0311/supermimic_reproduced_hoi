"""Common table action labels for rollout metrics and the evaluation dashboard."""

TABLE_ACTIONS = ("carry", "grasp table leg", "push", "kick", "pull")
TABLE_ACTION_LOG_EVERY = 50


def action_key(action):
    if action not in TABLE_ACTIONS:
        raise ValueError(f"Unknown table action label: {action}")
    return action.replace(" ", "_")


def episode_metrics(totals):
    """Summarize completed episodes, including terminal steps and timeouts."""
    metrics = {}
    for action, (episodes, failures, normalized_return_sum) in zip(TABLE_ACTIONS, totals, strict=True):
        prefix = f"tables_by_action/{action_key(action)}"
        metrics[f"{prefix}/episode_count"] = int(episodes)
        if episodes:
            metrics[f"{prefix}/failure_rate"] = failures / episodes
            metrics[f"{prefix}/normalized_return"] = normalized_return_sum / episodes
    return metrics
