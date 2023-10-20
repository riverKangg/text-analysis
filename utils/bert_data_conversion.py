import numpy as np


# Converting the lines to BERT format
def convert_lines(example, max_seq_length, tokenizer):
    """
    Convert a list of text examples into BERT input format.

    Args:
        example (list): A list of text examples to be converted.
        max_seq_length (int): Maximum sequence length for BERT input.
        tokenizer: The BERT tokenizer for tokenization.

    Returns:
        numpy.ndarray: An array of tokenized and padded sequences in BERT format.

    This function tokenizes and converts a list of text examples into BERT input format.
    It tokenizes the text, truncates or pads to the specified `max_seq_length`, and adds
    special tokens [CLS] and [SEP]. The resulting sequences are stored in a numpy array.

    It also prints the count of examples that were truncated due to exceeding `max_seq_length`.

    Example:
    ```
    sequences = convert_lines(text_examples, max_sequence_length, bert_tokenizer)
    ```
    """
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
                max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(f"Number of examples truncated: {longer}")
    return np.array(all_tokens)
