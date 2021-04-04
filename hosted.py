#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import things not specific to model replication first
import base64
import getpass
import os
import random
import sys

import colorama
import flask

colorama.init()

print("Loading TensorFlow in compatibility mode...")

import tensorflow.compat.v1 as tf_v1

tf_v1.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

GPU_ENABLED = tf_v1.test.is_gpu_available()

print("Running on " + (colorama.Fore.LIGHTGREEN_EX + "GPU" if GPU_ENABLED else colorama.Fore.LIGHTRED_EX + "CPU") + colorama.Fore.LIGHTYELLOW_EX + " mode.")

print("Loading generator and story management...")

from generator.gpt2.gpt2_generator import *
from story import grammars
from story.story_manager import *
from story.utils import *

print("Loading model...")

generator = GPT2Generator()
story_manager = UnconstrainedStoryManager(generator, upload_story=False, cloud=False)
story_manager.inference_timeout = 300

# Code

def get_curated_exposition(
    setting_key, character_key, name, character, setting_description
):
    name_token = "<NAME>"
    if (
        character_key == "noble"
        or character_key == "knight"
        or character_key == "wizard"
        or character_key == "peasant"
        or character_key == "rogue"
    ):
        context = grammars.generate(setting_key, character_key, "context") + "\n\n"
        context = context.replace(name_token, name)
        prompt = grammars.generate(setting_key, character_key, "prompt")
        prompt = prompt.replace(name_token, name)
    else:
        context = (
            "You are "
            + name
            + ", a "
            + character_key
            + " "
            + setting_description
            + "You have a "
            + character["item1"]
            + " and a "
            + character["item2"]
            + ". "
        )

        prompt = random.choice(character['prompts'].values())

    return context, prompt


def start_new_story():
    with open('story/story_data.yaml', 'r') as stream:
        story_data = yaml.safe_load(stream)

    # Select a random setting

    # Only fantasy is supported right now
    #setting_key = random.choice(list(story_data['settings'].keys()))
    setting_key = 'fantasy'

    character_key = random.choice(list(story_data['settings'][setting_key]['characters'].keys()))
    name = grammars.direct(setting_key, 'fantasy_name')
    character = story_data['settings'][setting_key]['characters'][character_key]
    setting_description = story_data['settings'][setting_key]['description']

    # Generate context and prompt
    context, prompt = get_curated_exposition(setting_key, character_key, name, character, setting_description)

    story_manager.generator.generate_num = 120
    story_manager.start_new_story(
        prompt, context=context, upload_story=False
    )
    story_manager.generator.generate_num = story_manager.generator.default_gen_num

    print(f'---- BEGIN STORY START ----\n{get_story_chunk()}\n---- END STORY START ----')


def feed_story_data(action: str):
    # Preprocess action
    if action == "":
        action = "\n> \n"

    elif action[0] != '"':
        action = action.strip()

        action = action[0].lower() + action[1:]

        if action[-1] not in [".", "?", "!"]:
            action = action + "."

        action = first_to_second_person(action)

        action = "\n> " + action + "\n"

    if "say" in action or "ask" in action or "\"" in action:
        story_manager.generator.generate_num = 120

    # Run model for action
    story_manager.act_with_timeout(action)

    if len(story_manager.story.results) >= 2:
        similarity = get_similarity(
            story_manager.story.results[-1], story_manager.story.results[-2]
        )

        # Don't commit repeating actions to try and stop the AI getting stuck in a loop
        if similarity > 0.9:
            story_manager.story.actions = story_manager.story.actions[:-1]
            story_manager.story.results = story_manager.story.results[:-1]

    story_manager.generator.generate_num = story_manager.generator.default_gen_num

    print(f'---- BEGIN STORY CHUNK ----\n{get_story_chunk()}\n---- END STORY CHUNK ----')


def get_story_chunk():
    if len(story_manager.story.results) == 0:
        return story_manager.story.story_start
    else:
        return f"{story_manager.story.actions[-1]}\n{story_manager.story.results[-1]}"


# Flask stuff

app = flask.Flask(__name__)

@app.route('/chunk', methods=['GET'])
def get_chunk():
    return flask.Response(
        json.dumps(get_story_chunk()),
        mimetype="application/json"
    )

@app.route('/act', methods=['POST'])
def do_action():
    action = flask.request.form.get('action')

    feed_story_data(action)

    return flask.Response(
        json.dumps(get_story_chunk()),
        mimetype="application/json"
    )

@app.route('/reset', methods=['GET'])
def reset_internal():
    start_new_story()

    return flask.Response(
        json.dumps(get_story_chunk()),
        mimetype="application/json"
    )

@app.route('/revert', methods=['GET'])
def revert_action():
    story_manager.story.actions = story_manager.story.actions[:-1]
    story_manager.story.results = story_manager.story.results[:-1]

    return flask.Response(
        json.dumps(get_story_chunk()),
        mimetype="application/json"
    )

@app.route('/retry', methods=['GET'])
def retry_action():
    last_action = story_manager.story.actions.pop()
    story_manager.story.results.pop()

    story_manager.act_with_timeout(last_action)

    return flask.Response(
        json.dumps(get_story_chunk()),
        mimetype="application/json"
    )

@app.route('/transcript', methods=['GET'])
def read_transcript():
    return flask.Response(
        json.dumps(str(story_manager.story)),
        mimetype="application/json"
    )


if __name__ == "__main__":
    print("Starting a story to preload BLAS...")
    start_new_story()

    print("Starting Flask")
    app.run('0.0.0.0', port=8181, debug=True, use_reloader=False)
