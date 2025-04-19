# listenlearn

Generate audio file for learning vocabulary from any text using text-to-speech.
It uses [edge-tts](https://github.com/rany2/edge-tts) for text-to-speech
that you can use without any additional steps.

## Configuration

You need to have the file `~/.local/share/listenlearn/config.json` with a content like the following:

```json
{
  "learning_modes": {
    "ensk": {
      "target_params": {"language": "sk"},
      "fallback_params": {"language": "en", "rate": 1.3, "pitch_delta_hz": 20}
    }
  },
  "default_learning_mode": "ensk",
  "openai_providers": {
    "pollinations": {
      "base_url": "https://text.pollinations.ai/openai",
      "model": "mistral",
      "authless": true
    }
  },
  "default_openai_provider": "pollinations"
}
```

Here I’ve used [pollinations](https://pollinations.ai/) that you can
use without any additional steps.

## Usage

1.  **Extract vocabulary using the `learningset` command:**

    This command extracts potential vocabulary from a text source and saves it to a CSV file.

    ```bash
    python main.py learningset -o vocabulary.csv < input.txt
    ```

    You can also skip this and manually create the `vocabulary.csv` file.
    The CSV shall have the following columns: `target_language`, `fallback_language`
    and then list the entries in the order they should be spoken.

2.  **(Optional) Refine the vocabulary:**

    The `learningset` command may generate entries that are not useful for learning
    (e.g., proper nouns, duplicates, already known words).
    Manually edit the `vocabulary.csv` file to remove these entries.

3.  **Generate audio using the `settts` command:**

    This command reads the vocabulary from the CSV file and generates a concatenated audio file with the target and fallback language pronunciations.

    ```bash
    python main.py settts -o vocabulary.mp3 vocabulary.csv
    ```

Extra: **`tts`:** Generates an audio file of the entire input text.

```bash
python main.py tts -o output.mp3 < input.txt
```

## Features

* **Repetition Scheme:** The generated audio repeats each vocabulary item in the
  following sequence: target language, fallback language, target language, target language.
  It also has a very basic queue so it will be repeated once after a few other items.
* **Sentence and Chunk Extraction:** The `learningset` command extracts sentences and
  short chunks from the input text using LLM ([instructor](https://python.useinstructor.com/)
  for the structured output). While sentence extraction generally provides good results,
  the quality of short chunk extraction isn’t very good
  (though still a good starting point IMO).
  The sentence is added after all its chunks.
