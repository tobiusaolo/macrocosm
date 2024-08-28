import math
from typing import List

# TODO: Remove
import bittensor as bt

from taoverse.model.competition.data import EpsilonFunc
from taoverse.model.data import EvalResult


def should_retry_model(
    epsilon_func: EpsilonFunc, curr_block: int, eval_history: List[EvalResult]
) -> bool:
    """Determines if a model should be retried based on its evaluation history and the current state.

    A model is retryable if any of the following apply:
        - It has never been evaluated.
        - When it was last evaluated it had a better loss than the top model but couldn't overcome the epsilon disadvantage.
          However, now epsilon has lowered to the point that it may be able to overcome the epsilon disadvantage.
        - The model has only been evaluated once and it hit an error. In this case, we allow a single retry.

    Args:
        epsilon_func (EpsilonFunc): The function to compute the current epsilon.
        curr_block (int): The current block
        eval_history (List[EvalResult]): The (potentially empty) evaluation history of the model.
    """
    # TODO: Remove logging from below.
    # If the model has never been evaluated, we should retry it.
    if not eval_history:
        bt.logging.trace("Model has never been evaluated. Retrying.")
        return True

    # Find the most recent successful eval.
    last_successful_eval = None
    for eval_result in reversed(eval_history):
        if eval_result.score != math.inf:
            last_successful_eval = eval_result
            break

    if last_successful_eval:
        bt.logging.trace(f"Last successul eval: {last_successful_eval}.")
        # If this model had worse loss than the top model during the last eval, no need to retry.
        # NOTE: "score" = avg_loss so lower is better.
        if last_successful_eval.score > last_successful_eval.winning_model_score:
            bt.logging.trace(f"Score is worse. Returning false.")
            return False

        # Otherwise, this model is potentially better than the top model but at the time it was evaluated
        # it couldn't overcome the epsilon disadvantage. Check if epsilon has changed to the point where
        # we should retry this model now.
        curr_epsilon = epsilon_func.compute_epsilon(
            current_block=curr_block,
            model_block=last_successful_eval.winning_model_block,
        )
        # Compute the adjusted loss of the top model based on the current epsilon.
        top_model_score = last_successful_eval.winning_model_score * (1 - curr_epsilon)
        bt.logging.trace(
            f"Epsilon is {curr_epsilon}. top_model_adj_loss is {top_model_score}. My loss = {last_successful_eval.score}."
        )
        return last_successful_eval.score < top_model_score

    # This model has been evaluated but has errored every time. Allow a single retry in this case.
    bt.logging.trace(f"No successful evals. Returning {len(eval_history) < 2}.")
    return len(eval_history) < 2
