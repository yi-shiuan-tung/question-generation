from preprocess import extract_sentences, get_words


def test_extract_sentences():
    paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore" \
                " et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut" \
                " aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse" \
                " cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in" \
                " culpa qui officia deserunt mollit anim id est laborum."
    indices, sentences = extract_sentences(paragraph)
    assert len(sentences) == 4
    assert len(indices) == len(sentences)
    for i in range(len(indices)):
        assert paragraph[indices[i]] == sentences[i][0]


def test_get_words():
    string = "hello world. Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born " \
             "September 4, 1981)."
    assert len(get_words(string)) == 12

    string = "hello world.test"
    assert len(get_words(string)) == 3

    assert len(get_words("")) == 0
