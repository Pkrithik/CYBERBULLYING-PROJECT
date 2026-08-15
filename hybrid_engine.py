import re
import nltk
import os

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Load bad words once globally
BAD_WORDS = set()
bad_words_path = os.path.join(os.path.dirname(__file__), "bad_words.txt")
if os.path.exists(bad_words_path):
    with open(bad_words_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                BAD_WORDS.add(line)

def preprocess_text(text):
    """
    Centralized text cleaning used for both training and inference.
    """
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"@[\w]*", "", text)
    text = re.sub(r"[^a-zA-Z#]", " ", text)
    
    tokens = text.split()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    
    return " ".join(lemmatized)

def detect_toxic_words(text):
    """
    Detect explicit toxic words using token-aware matching.
    """
    if not isinstance(text, str):
        return False
        
    # Simple tokenization by word boundaries
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Check intersection
    return bool(tokens.intersection(BAD_WORDS))

def detect_targeted_abuse(text):
    """
    Detects whether abusive/offensive language is specifically
    directed toward another person.

    A pronoun such as "you" alone is NOT considered abuse.
    Targeted abuse requires:
        1. A target indicator (you, your, u r, etc.)
        AND
        2. An abusive/insulting expression.
    """

    if not isinstance(text, str):
        return False

    text_lower = text.lower().strip()

    # ---------------------------------------------------------
    # 1. Explicit phrases that are clearly targeted abuse
    # ---------------------------------------------------------
    explicit_targeted_patterns = [
        r'\bkill yourself\b',
        r'\bkys\b',
        r'\bgo die\b',
        r'\bdie already\b',
        r'\bshut up\b',
        r'\bget lost\b',
        r'\bnobody likes you\b',
        r'\bno one likes you\b',
        r'\byou are worthless\b',
        r'\byou are useless\b',
        r'\byou are disgusting\b',
        r'\byou are pathetic\b',
        r'\byou are horrible\b',
    ]

    for pattern in explicit_targeted_patterns:
        if re.search(pattern, text_lower):
            return True

    # ---------------------------------------------------------
    # 2. Target indicators
    # ---------------------------------------------------------
    target_patterns = [
        r'\byou\b',
        r'\byour\b',
        r'\bu r\b',
        r'\bur\b',
        r'\byou\'re\b',
        r'\byouve\b'
    ]

    has_target = any(
        re.search(pattern, text_lower)
        for pattern in target_patterns
    )

    if not has_target:
        return False

    # ---------------------------------------------------------
    # 3. Words that represent insults / abusive descriptions
    # ---------------------------------------------------------
    abusive_descriptors = {
        "ugly",
        "stupid",
        "idiot",
        "dumb",
        "moron",
        "loser",
        "worthless",
        "useless",
        "pathetic",
        "disgusting",
        "horrible",
        "awful",
        "trash",
        "fool",
        "jerk",
        "bitch",
        "asshole",
        "fat",
        "skinny",
        "ugliest",
        "disgrace",
        "failure"
    }

    tokens = set(re.findall(r'\b[a-zA-Z]+\b', text_lower))

    has_abusive_descriptor = bool(
        tokens.intersection(abusive_descriptors)
    )

    # ---------------------------------------------------------
    # 4. Target + abusive descriptor = targeted abuse
    # ---------------------------------------------------------
    if has_target and has_abusive_descriptor:
        return True

    return False


def detect_body_shaming(text):
    """
    Detect common body-shaming expressions.
    This is a rule-based component and works together
    with the ML classifier.
    """

    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    body_shaming_patterns = [

        # Weight / body size
        r'\byou are fat\b',
        r"\byou're fat\b",
        r'\byou look fat\b',
        r'\btoo fat\b',
        r'\bso fat\b',
        r'\bvery fat\b',

        r'\byou are obese\b',
        r"\byou're obese\b",

        r'\byou are overweight\b',
        r"\byou're overweight\b",

        r'\byou are too skinny\b',
        r"\byou're too skinny\b",
        r'\btoo skinny\b',
        r'\bso skinny\b',
        r'\byou are so thin\b',
        r'\byou are so lean\b',

        r'\byou are thin\b',
        r"\byou're too thin\b",

        # Appearance / animal comparison insults
        r'\byou look like an elephant\b',
        r'\byou look like a pig\b',
        r'\byou look like a whale\b',
        r'\byou look like a cow\b',
        r'\byou look like a monkey\b',
        r'\byou look like a dog\b',
         
                

        # General appearance insults
        r'\byou look disgusting\b',
        r'\byou look horrible\b',
        r'\byou look terrible\b',
        r'\byou look gross\b',    r'\byou are fat\b',
        r'\byoure fat\b',
        r'\byou look fat\b',
        r'\byou look so fat\b',
        r'\byou look really fat\b',
        r'\byou look very fat\b',
        r'\byou look fatter\b',
        r'\byou have gotten fat\b',
        r'\byou got fat\b',
        r'\byou got so fat\b',
        r'\byou have become fat\b',
        r'\byou are getting fat\b',
        r'\byou are too fat\b',
        r'\byou are extremely fat\b',
        r'\byou are seriously fat\b',
        r'\byou are overweight\b',
        r'\byoure overweight\b',
        r'\byou look overweight\b',
        r'\byou are obese\b',
        r'\byou look obese\b',

        # Natural conversational phrases
        r'\bhow did you get so fat\b',
        r'\bwhen did you get this fat\b',
        r'\bwhy did you get so fat\b',
        r'\bhave you gained weight\b',
        r'\byou have gained so much weight\b',
        r'\byou gained a lot of weight\b',
        r'\byou really need to lose weight\b',
        r'\byou seriously need to lose weight\b',
        r'\byou need to lose some weight\b',
        r'\byou should lose some weight\b',
        r'\byou should go on a diet\b',
        r'\byou need a diet\b',

        # =========================================================
        # THIN / SKINNY
        # =========================================================

        r'\byou are skinny\b',
        r'\byoure skinny\b',
        r'\byou look skinny\b',
        r'\byou look so skinny\b',
        r'\byou look really skinny\b',
        r'\byou are too skinny\b',
        r'\byou are extremely skinny\b',
        r'\byou are ridiculously skinny\b',
        r'\byou are painfully skinny\b',
        r'\byou look dangerously skinny\b',
        r'\byou are too thin\b',
        r'\byou look too thin\b',
        r'\byou are extremely thin\b',
        r'\byou look extremely thin\b',
        r'\byou look like a stick\b',
        r'\byou look like a twig\b',
        r'\byou look like a toothpick\b',
        r'\byou are nothing but skin and bones\b',
        r'\byou are all skin and bones\b',

        # Natural conversational phrases
        r'\bdo you even eat\b',
        r'\bdo you ever eat\b',
        r'\bwhen was the last time you ate\b',
        r'\byou need to eat more\b',
        r'\byou look like you havent eaten\b',
        r'\byou look like you havent eaten in days\b',

        # =========================================================
        # BODY SIZE / SHAPE
        # =========================================================

        r'\byour body is huge\b',
        r'\byour body is massive\b',
        r'\byour body is too big\b',
        r'\byour body is too small\b',
        r'\byour body looks weird\b',
        r'\byour body looks terrible\b',
        r'\byour body looks awful\b',
        r'\byou have a huge body\b',
        r'\byou have a tiny body\b',
        r'\byou have no shape\b',
        r'\byour body has no shape\b',
        r'\byou have a weird body\b',
        r'\byou have an awkward body\b',

        # =========================================================
        # BELLY / STOMACH
        # =========================================================

        r'\byour stomach is huge\b',
        r'\byour stomach is massive\b',
        r'\byour stomach is too big\b',
        r'\byou have a huge stomach\b',
        r'\byou have a big stomach\b',
        r'\byou have a massive stomach\b',
        r'\byour belly is huge\b',
        r'\byour belly is massive\b',
        r'\byour belly is too big\b',
        r'\byou have a huge belly\b',
        r'\byou have a big belly\b',
        r'\byou have a massive belly\b',
        r'\byour belly is sticking out\b',
        r'\byour stomach is sticking out\b',

        # Conversational / mocking
        r'\blook at that belly\b',
        r'\blook at your belly\b',
        r'\bwhat happened to your stomach\b',
        r'\bwhat happened to your belly\b',
        r'\bthat belly is huge\b',

        # =========================================================
        # FACE
        # =========================================================

        r'\byour face is ugly\b',
        r'\byour face looks ugly\b',
        r'\byou have an ugly face\b',
        r'\byour face looks weird\b',
        r'\byour face looks strange\b',
        r'\byour face looks terrible\b',
        r'\byour face looks awful\b',
        r'\byou have a weird face\b',
        r'\byou have a strange face\b',
        r'\byou have a horrible face\b',
        r'\byour face is too big\b',
        r'\byour face is too small\b',
        r'\byour face is huge\b',

        # =========================================================
        # NOSE
        # =========================================================

        r'\byour nose is huge\b',
        r'\byour nose is massive\b',
        r'\byour nose is too big\b',
        r'\byou have a huge nose\b',
        r'\byou have a massive nose\b',
        r'\byou have a big nose\b',
        r'\byour nose looks weird\b',
        r'\byour nose looks awful\b',
        r'\byour nose is ugly\b',

        # =========================================================
        # TEETH / SMILE
        # =========================================================

        r'\byour teeth are ugly\b',
        r'\byour teeth look terrible\b',
        r'\byour teeth look horrible\b',
        r'\byou have ugly teeth\b',
        r'\byou have terrible teeth\b',
        r'\byou have horrible teeth\b',
        r'\byour teeth are disgusting\b',
        r'\byour smile is ugly\b',
        r'\byour smile looks terrible\b',

        # =========================================================
        # LEGS / ARMS
        # =========================================================

        r'\byour legs are too skinny\b',
        r'\byour legs are extremely skinny\b',
        r'\byou have skinny legs\b',
        r'\byou have tiny legs\b',
        r'\byour legs look like sticks\b',

        r'\byour arms are too skinny\b',
        r'\byour arms are extremely skinny\b',
        r'\byou have skinny arms\b',
        r'\byou have tiny arms\b',
        r'\byour arms look like sticks\b',

        # =========================================================
        # HEIGHT
        # =========================================================

        r'\byou are so short\b',
        r'\byou are really short\b',
        r'\byou are extremely short\b',
        r'\byou are ridiculously short\b',
        r'\byou look tiny\b',
        r'\byou are tiny\b',
        r'\byou are too short\b',
        r'\byou are short for your age\b',
        r'\byou look like a little kid\b',
        r'\byou look like a child\b',

        r'\byou are too tall\b',
        r'\byou are ridiculously tall\b',
        r'\byou are extremely tall\b',
        r'\byou look like a giant\b',
        r'\byou look like a walking skyscraper\b',

        # =========================================================
        # BODY COMPARISONS
        # =========================================================

        r'\byou look like a pig\b',
        r'\byou look like an elephant\b',
        r'\byou look like a whale\b',
        r'\byou look like a hippo\b',
        r'\byou look like a cow\b',
        r'\byou look like a buffalo\b',
        r'\byou look like a balloon\b',
        r'\byou look like a potato\b',
        r'\byou look like a sack of potatoes\b',

        r'\byou look like a stick\b',
        r'\byou look like a twig\b',
        r'\byou look like a toothpick\b',
        r'\byou look like a skeleton\b',
        r'\byou look like a walking skeleton\b',

        r'\byou look like a giant\b',
        r'\byou look like a little kid\b',
        r'\byou look like a child\b',

        # =========================================================
        # CLOTHING / BODY FIT
        # =========================================================

        r'\bthose clothes dont fit you\b',
        r'\bthose clothes do not fit you\b',
        r'\bthat shirt is too tight for you\b',
        r'\byou are too big for that shirt\b',
        r'\byou are too big for those clothes\b',
        r'\byou are too small for that dress\b',
        r'\byou should wear something that hides your body\b',
        r'\byou should wear something loose\b',
        r'\bthat outfit makes you look fat\b',
        r'\bthat dress makes you look fat\b',
        r'\bthat shirt makes you look fat\b',

        # =========================================================
        # APPEARANCE + BODY
        # =========================================================

        r'\byou are ugly\b',
        r'\byoure ugly\b',
        r'\byou look ugly\b',
        r'\byou look so ugly\b',
        r'\byou look really ugly\b',
        r'\byou look terrible\b',
        r'\byou look horrible\b',
        r'\byou look awful\b',
        r'\byou look disgusting\b',
        r'\byou are unattractive\b',
        r'\byou look unattractive\b',

        # =========================================================
        # MOCKING / SARCASTIC BODY COMMENTS
        # =========================================================

        r'\bwhat happened to your body\b',
        r'\bwhat happened to your face\b',
        r'\bwhat happened to your appearance\b',
        r'\bis that really your body\b',
        r'\bis that really your face\b',
        r'\bdid you always look like that\b',
        r'\bhave you always looked like that\b',
        r'\bwho told you that you look good\b',
        r'\bwho told you that outfit looks good\b',
        r'\bthat outfit is not helping you\b',
        r'\bthat outfit does nothing for you\b',
        r'\byou should not wear that\b',
        r'\byou should not be wearing that\b',
        r'\bwhy would you wear that\b',

        # =========================================================
        # NATURAL INSULTING PHRASES
        # =========================================================

        r'\bhonestly you look terrible\b',
        r'\bhonestly you look awful\b',
        r'\bhonestly you look ugly\b',
        r'\bnot gonna lie you look fat\b',
        r'\bnot gonna lie youre fat\b',
        r'\bnot gonna lie you look skinny\b',
        r'\bno offense but you look fat\b',
        r'\bno offense but youre fat\b',
        r'\bno offense but you are ugly\b',
        r'\bno offense but you look ugly\b',
        r'\bim just being honest you look fat\b',
        r'\bim just being honest you look ugly\b',
        r'\bjust being honest you look terrible\b',
        r'\bjust being honest you need to lose weight\b',

        # =========================================================
        # MOCKING BODY-SHAME STYLE PHRASES
        # =========================================================

        r'\bbetter stay away from the buffet\b',
        r'\bstay away from the food\b',
        r'\bstop eating so much\b',
        r'\bstop eating everything\b',
        r'\bmaybe eat less\b',
        r'\bmaybe try eating less\b',
        r'\bput the food down\b',
        r'\bput that food down\b',
        r'\bthe gym needs to see you\b',
        r'\byou need to see a gym\b',
        r'\bthe mirror must hate you\b',
        r'\bthe mirror is not your friend\b',
    

        # Body-related comparisons
        r'\byou look like a whale\b',
        r'\byou look like a balloon\b',
        r'\byou look like a skeleton\b',
        r'\byou look like a stick\b',

        # Appearance
        r'\byou are ugly\b',
        r"\byou're ugly\b",
        r'\byou look ugly\b',

        r'\byou look disgusting\b',
        r'\byour body is disgusting\b',

        # Body parts
        r'\byour stomach is huge\b',
        r'\byour belly is huge\b',
        r'\byour stomach is big\b',
        r'\byour belly is big\b',

        # Weight-related insults
        r'\blose some weight\b',
        r'\bneed to lose weight\b',
        r'\bgo lose weight\b',
        r'\blook at your weight\b',

        # Appearance insults
        r'\bdisgusting body\b',
        r'\bugly body\b',
        r'\bfat body\b',
        r'\bskinny body\b'
    ]

    for pattern in body_shaming_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def hybrid_decision(ml_prediction, text):
    """
    Hybrid cyberbullying detection.

    Returns exactly 7 values:

    1. prediction
    2. label
    3. status
    4. message
    5. toxic_word_detected
    6. targeted_abuse
    7. body_shaming
    """

    is_toxic = detect_toxic_words(text)
    is_targeted = detect_targeted_abuse(text)
    is_body_shaming = detect_body_shaming(text)

    # ------------------------------------------------
    # CASE 1: ML identifies cyberbullying
    # ------------------------------------------------

    if ml_prediction == 1:
        return (
            1,
            "cyberbullying",
            "blocked",
            "Cyberbullying was detected by the machine-learning classifier.",
            is_toxic,
            is_targeted,
            is_body_shaming
        )

    # ------------------------------------------------
    # CASE 2: Body shaming + targeted person
    # ------------------------------------------------

    if is_body_shaming and is_targeted:
        return (
            1,
            "body_shaming",
            "blocked",
            "Body-shaming content targeting another person was detected.",
            is_toxic,
            is_targeted,
            is_body_shaming
        )

    # ------------------------------------------------
    # CASE 3: Toxic word + targeted person
    # ------------------------------------------------

    if is_toxic and is_targeted:
        return (
            1,
            "cyberbullying",
            "blocked",
            "Cyberbullying detected: abusive language targeting another person was found.",
            is_toxic,
            is_targeted,
            is_body_shaming
        )

    # ------------------------------------------------
    # CASE 4: Body shaming but no clear target
    # ------------------------------------------------

    if is_body_shaming and not is_targeted:
        return (
            0,
            "review",
            "review",
            "Possible body-shaming content detected. The message requires contextual review.",
            is_toxic,
            is_targeted,
            is_body_shaming
        )

    # ------------------------------------------------
    # CASE 5: Offensive word but not targeted
    # ------------------------------------------------

    if is_toxic and not is_targeted:
        return (
            0,
            "review",
            "review",
            "Offensive language was detected, but the message does not clearly target another person.",
            is_toxic,
            is_targeted,
            is_body_shaming
        )

    # ------------------------------------------------
    # CASE 6: Safe
    # ------------------------------------------------

    return (
        0,
        "safe",
        "allowed",
        "No cyberbullying detected.",
        is_toxic,
        is_targeted,
        is_body_shaming
    )