use std::borrow::Cow;

use regex::Regex;
use unidecode::unidecode;

lazy_static! {
    static ref URL_RE: Regex = Regex::new(r"https?://\S+").unwrap();
    static ref MENTION_RE: Regex = Regex::new(r"@\S+").unwrap();
    static ref WORD_RE: Regex = Regex::new(r"\b\w\w+\b").unwrap();
}

fn reduce_lengthening(string: &str) -> String {
    let mut output: String = String::new();

    let mut counter = 0;
    let mut last_char: Option<char> = None;

    for c in string.chars() {
        match last_char {
            Some(last) => {
                if c == last {
                    counter += 1;
                } else {
                    counter = 0;
                    last_char = Some(c);
                }
            }
            None => {
                last_char = Some(c);
            }
        }

        if counter < 3 {
            output.push(c);
        }
    }

    output
}

#[test]
fn test_reduce_lengthening() {
    assert_eq!(reduce_lengthening("cool"), "cool");
    assert_eq!(reduce_lengthening("coool"), "coool");
    assert_eq!(reduce_lengthening("cooooool"), "coool");
    assert_eq!(reduce_lengthening("cooooooooooolool"), "cooolool");
}

fn strip_urls(string: &str) -> Cow<str> {
    return URL_RE.replace_all(string, "");
}

#[test]
fn test_strip_urls() {
    assert_eq!(
        strip_urls("This is a url: http://lemonde.fr no?"),
        "This is a url:  no?"
    )
}

fn strip_mentions(string: &str) -> Cow<str> {
    return MENTION_RE.replace_all(string, "");
}

#[test]
fn test_strip_mentions() {
    assert_eq!(
        strip_mentions("This is a mention: @Yomgui no?"),
        "This is a mention:  no?"
    )
}

fn tokenize(string: &str) -> Vec<String> {
    let string = strip_urls(string);
    let string = strip_mentions(&string);
    let string = unidecode(&string);

    WORD_RE
        .find_iter(&string)
        .map(|word| reduce_lengthening(word.as_str()).to_lowercase())
        .collect()
}

#[test]
fn test_tokenize() {
    assert_eq!(
        tokenize("Hello World, this is I the élémental @Yomgui http://lemonde.fr type looooool! #Whatever"),
        vec!["hello", "world", "this", "is", "the", "elemental", "type", "loool", "whatever"]
    );
}
