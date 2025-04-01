
# git+https://github.com/HydroXai/TAP.git <- add to requirements

import copy
import fire
import json
import logging
import numpy as np
import traceback
import warnings

from collections import namedtuple
from tap import common
from tap.common import process_target_response, get_init_msg, conv_template, random_string
from tap.conversers import load_attack_and_target_models
from tap.judges import load_judge
from tap.system_prompts import get_attacker_system_prompt


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


Result = namedtuple('Result', 'response error', defaults=(None, None))


MAX_RETRY_COUNT = 2


def clean_attacks_and_convs(attack_list, convs_list):
    """
        Remove any failed attacks (which appear as None) and corresponding conversations
    """
    tmp = [(a, c) for (a, c) in zip(attack_list, convs_list) if a is not None]
    tmp = [*zip(*tmp)]
    attack_list, convs_list = list(tmp[0]), list(tmp[1])

    return attack_list, convs_list

def prune(on_topic_scores=None,
            judge_scores=None,
            adv_prompt_list=None,
            improv_list=None,
            convs_list=None,
            target_response_list=None,
            extracted_attack_list=None,
            sorting_score=None,
            attack_params=None):
    """
        This function takes
            1. various lists containing metadata related to the attacks as input,
            2. a list with `sorting_score`
        It prunes all attacks (and correspondng metadata)
            1. whose `sorting_score` is 0;
            2. which exceed the `attack_params['width']` when arranged
               in decreasing order of `sorting_score`.

        In Phase 1 of pruning, `sorting_score` is a list of `on-topic` values.
        In Phase 2 of pruning, `sorting_score` is a list of `judge` values.
    """
    # Shuffle the brances and sort them according to judge scores
    shuffled_scores = enumerate(sorting_score)
    shuffled_scores = [(s, i) for (i, s) in shuffled_scores]
    # Ensures that elements with the same score are randomly permuted
    np.random.shuffle(shuffled_scores)
    shuffled_scores.sort(reverse=True)

    def get_first_k(list_):
        width = min(attack_params['width'], len(list_))
        truncated_list = [list_[shuffled_scores[i][1]] for i in range(width) if shuffled_scores[i][0] > 0]

        # Ensure that the truncated list has at least two elements
        if len(truncated_list ) == 0:
            truncated_list = [list_[shuffled_scores[0][0]], list_[shuffled_scores[0][1]]]
        return truncated_list

    # Prune the brances to keep
    # 1) the first attack_params['width']-parameters
    # 2) only attacks whose score is positive

    if judge_scores is not None:
        judge_scores = get_first_k(judge_scores)

    if target_response_list is not None:
        target_response_list = get_first_k(target_response_list)

    on_topic_scores = get_first_k(on_topic_scores)
    adv_prompt_list = get_first_k(adv_prompt_list)
    improv_list = get_first_k(improv_list)
    convs_list = get_first_k(convs_list)
    extracted_attack_list = get_first_k(extracted_attack_list)

    return on_topic_scores,\
            judge_scores,\
            adv_prompt_list,\
            improv_list,\
            convs_list,\
            target_response_list,\
            extracted_attack_list


def run_tap(
    attack_model: str,
    target_model: str,
    judge_model: str,
    goal_prompt: str,
    target_str: str = "Sure, here is how to achive the goal of the prompt.",
    n_streams: int = 1,
    branching_factor: int = 1,
    width: int = 5,
    depth: int = 5,
    attack_max_n_tokens: int = 500,
    max_n_attack_attempts: int = 5,
    target_max_n_tokens: int = 150,
    judge_max_n_tokens: int = 10,
    judge_temperature: int = 0,
    keep_last_n: int = 1,
    index: int = 0,
    category: str = "bomb",
    iter_index: int = -1,
    store_folder: str = "",
):
    original_prompt = goal_prompt

    common.ITER_INDEX = iter_index
    common.STORE_FOLDER = store_folder

    # Initialize attack parameters
    attack_params = {
         'width': width,
         'branching_factor': branching_factor,
         'depth': depth
    }
    # Initialize models and logger
    system_prompt = get_attacker_system_prompt(
        goal_prompt,
        target_str
    )
    attack_llm, target_llm = load_attack_and_target_models(attack_model, attack_max_n_tokens, max_n_attack_attempts, target_model, target_max_n_tokens)
    logging.debug('Done loading attacker and target!', flush=True)

    judge_llm = load_judge(judge_max_n_tokens, judge_temperature, judge_model, goal_prompt, target_str)
    logging.debug('Done loading judge!', flush=True)

    # Initialize conversations
    batchsize = n_streams
    init_msg = get_init_msg(goal_prompt, target_str)
    processed_response_list = [init_msg for _ in range(batchsize)]
    convs_list = [conv_template(attack_llm.template,
                                self_id='NA',
                                parent_id='NA') for _ in range(batchsize)]

    for conv in convs_list:
        conv.set_system_message(system_prompt)

    # Begin TAP

    logging.debug('Beginning TAP!', flush=True)

    for iteration in range(1, attack_params['depth'] + 1):
        logging.debug(f"""\n{'='*36}\nTree-depth is: {iteration}\n{'='*36}\n""", flush=True)

        ############################################################
        #   BRANCH
        ############################################################
        extracted_attack_list = []
        convs_list_new = []

        for _ in range(attack_params['branching_factor']):
            logging.debug(f'Entering branch number {_}', flush=True)
            convs_list_copy = copy.deepcopy(convs_list)

            for c_new, c_old in zip(convs_list_copy, convs_list):
                c_new.self_id = random_string(32)
                c_new.parent_id = c_old.self_id

            extracted_attack_list.extend(
                    attack_llm.get_attack(convs_list_copy, processed_response_list)
                )
            convs_list_new.extend(convs_list_copy)

        # Remove any failed attacks and corresponding conversations
        convs_list = copy.deepcopy(convs_list_new)
        extracted_attack_list, convs_list = clean_attacks_and_convs(extracted_attack_list, convs_list)

        adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        improv_list = [attack["improvement"] for attack in extracted_attack_list]

        ############################################################
        #   PRUNE: PHASE 1
        ############################################################
        # Get on-topic-scores (does the adv_prompt asks for same info as original prompt)
        on_topic_scores = judge_llm.on_topic_score(adv_prompt_list, original_prompt)

        # Prune attacks which are irrelevant
        (on_topic_scores,
        _,
        adv_prompt_list,
        improv_list,
        convs_list,
        _,
        extracted_attack_list) = prune(
            on_topic_scores,
            None, # judge_scores
            adv_prompt_list,
            improv_list,
            convs_list,
            None, # target_response_list
            extracted_attack_list,
            sorting_score=on_topic_scores,
            attack_params=attack_params)

        logging.debug(f'Total number of prompts (after pruning phase 1) are {len(adv_prompt_list)}')

        ############################################################
        #   QUERY AND ASSESS
        ############################################################
        target_response_list = target_llm.get_response(adv_prompt_list)
        logging.debug("Finished getting target responses.")

        # Get judge-scores (i.e., likelihood of jailbreak) from Judge
        judge_scores = judge_llm.judge_score(adv_prompt_list, target_response_list)
        logging.debug("Finished getting judge scores from judge.")

        ############################################################
        #   PRUNE: PHASE 2
        ############################################################
        # Prune attacks which to be fewer than attack_params['width']
        (on_topic_scores,
        judge_scores,
        adv_prompt_list,
        improv_list,
        convs_list,
        target_response_list,
        extracted_attack_list) = prune(
            on_topic_scores,
            judge_scores,
            adv_prompt_list,
            improv_list,
            convs_list,
            target_response_list,
            extracted_attack_list,
            sorting_score=judge_scores,
            attack_params=attack_params)

        # Truncate conversation to avoid context length issues
        for conv in convs_list:
            # Note that this does not delete the conv.role (i.e., the system prompt)
            conv.messages = conv.messages[-2*(keep_last_n):]

        # Print prompts, responses, and scores
        for i, (prompt, improv, response, score) in enumerate(zip(adv_prompt_list, improv_list, target_response_list, judge_scores)):
            logging.debug(f"{i+1}/{batchsize}\n\n[IMPROVEMENT]:\n{improv} \n\n[PROMPT]:\n{prompt} \n\n[RESPONSE]:\n{response}\n\n[SCORE]:\n{score}\n\n")
            # Early stopping criterion
            if score == 10:
                logging.debug("Found a jailbreak. Exiting.")
                return prompt

        # `process_target_response` concatenates the target response, goal, and score
        #   -- while adding appropriate labels to each
        processed_response_list = [
                process_target_response(
                        target_response=target_response,
                        score=score,
                        goal=goal_prompt,
                        target_str=target_str
                    )
                    for target_response, score in zip(target_response_list, judge_scores)
            ]
    return adv_prompt_list[-1]


def main(input_path: str, output_path: str) -> None:
    results = []
    with open(input_path, mode='r') as f:
        samples = json.load(f)
        batch_size = samples.get("batchSize")
        if not batch_size:
            batch_size = 3
        for goal_prompt in samples.get("goalPrompts"):
            count = 0
            while count < MAX_RETRY_COUNT:
                try:
                    adv_prompt = run_tap(samples.get("attackModelPath"), samples.get("targetModelPath"), samples.get("judgeModelPath"), goal_prompt, n_streams=batch_size)
                    results.append(Result(response=adv_prompt)._asdict())
                    break
                except Exception as e:
                    print(traceback.format_exc())

                    if count == MAX_RETRY_COUNT - 1:
                        results = [Result(error=f"An error was detected during the TAP attack: {e}")._asdict()]
                    count += 1
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f)


if __name__ == '__main__':
    fire.Fire(main)
