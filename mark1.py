# import serial
# import sys
# import sounddevice as sd
# import queue
# from vosk import Model, KaldiRecognizer
# import json
# import re
# import spacy
# import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # =====================
# # SETTINGS
# # =====================
# ARDUINO_PORT = "COM8"
# BAUD_RATE = 9600
# MODEL_PATH = "models/vosk-model-small-en-us-0.15"
# DEFAULT_SPEED = 25  # degrees per step unless overridden
# SIMILARITY_THRESHOLD = 0.6  # cosine similarity threshold for servo detection

# # =====================
# # NUMBER WORDS CONVERSION
# # =====================
# NUM_WORDS = {
#     'zero': 0, 'one': 1, 'two': 2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
#     'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14,
#     'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19,
#     'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70,
#     'eighty':80, 'ninety':90, 'hundred':100
# }

# def text2num(text):
#     """
#     Parse number words from text and return integer.
#     Supports multi-word numbers like 'one hundred twenty three'.
#     Returns None if no number found.
#     """
#     tokens = text.lower().split()
#     total = 0
#     current = 0
#     found = False

#     for token in tokens:
#         if token not in NUM_WORDS:
#             # If we found a number phrase and hit a non-number word, stop parsing
#             if found:
#                 break
#             else:
#                 continue

#         found = True
#         val = NUM_WORDS[token]

#         if val == 100:
#             if current == 0:
#                 current = 1
#             current *= 100
#         else:
#             current += val

#     total = total + current

#     if found:
#         return total
#     else:
#         # No number words found, fallback to regex digits
#         nums = re.findall(r'\b\d+\b', text)
#         if nums:
#             return int(nums[0])
#         return None

# # =====================
# # INIT
# # =====================
# try:
#     arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
#     print(f"Connected to Arduino on {ARDUINO_PORT}")
# except serial.SerialException as e:
#     print(f"Error connecting to Arduino: {e}")
#     sys.exit(1)

# q = queue.Queue()
# model = Model(MODEL_PATH)
# rec = KaldiRecognizer(model, 16000)
# nlp = spacy.load("en_core_web_sm")

# servo_names = ["base", "arm", "claw"]
# servo_ids = {"base":"b", "arm":"s", "claw":"c"}

# vectorizer = TfidfVectorizer().fit(servo_names)

# def fuzzy_servo_match(word):
#     word_vec = vectorizer.transform([word])
#     sims = cosine_similarity(word_vec, vectorizer.transform(servo_names)).flatten()
#     max_idx = sims.argmax()
#     max_sim = sims[max_idx]
#     if max_sim >= SIMILARITY_THRESHOLD:
#         return servo_ids[servo_names[max_idx]]
#     return None

# def parse_command(text):
#     text = text.lower()
#     doc = nlp(text)

#     servo_id = None
#     for token in doc:
#         candidate = fuzzy_servo_match(token.text)
#         if candidate:
#             servo_id = candidate
#             break

#     query_phrases = ["what is", "current position", "position of", "where is", "show position", "get position", "gimme position", "position"]
#     is_query = any(phrase in text for phrase in query_phrases)

#     if is_query:
#         if servo_id:
#             return ("query", servo_id, None)
#         else:
#             return ("query", None, None)

#     speed = DEFAULT_SPEED
#     speed_match = re.search(r'speed\s+(\d+)', text)
#     if speed_match:
#         speed = int(speed_match.group(1))

#     angle = text2num(text)
#     if angle is None:
#         angle = 90

#     if servo_id is None:
#         print(f"Could not detect servo in '{text}'")
#         return None

#     return (servo_id, angle, speed)

# def send_servo_command(servo_id, angle, speed):
#     cmd = f"{servo_id} {angle} {speed}\n"
#     arduino.write(cmd.encode())
#     print(f"Sent: {cmd.strip()}")

# def query_servo_position(servo_id=None):
#     if servo_id is None:
#         cmd = "getpos\n"
#     else:
#         cmd = f"getpos {servo_id}\n"
#     arduino.write(cmd.encode())

#     start_time = time.time()
#     while True:
#         if arduino.in_waiting:
#             response = arduino.readline().decode().strip()
#             if response:
#                 return response
#         if time.time() - start_time > 2:
#             return None

# def interpret_commands(text):
#     steps = re.split(r'\band then\b|\bthen\b|\band\b', text.lower())
#     return [step.strip() for step in steps if step.strip()]

# print("Listening for multi-step commands...")

# with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
#                        channels=1, callback=lambda indata, frames, time_info, status: q.put(bytes(indata))):
#     while True:
#         data = q.get()
#         if rec.AcceptWaveform(data):
#             result = json.loads(rec.Result())
#             if "text" in result and result["text"].strip():
#                 heard = result["text"]
#                 print(f"Heard: {heard}")
#                 for step in interpret_commands(heard):
#                     parsed = parse_command(step)
#                     if parsed:
#                         servo_id, angle, speed = parsed
#                         if servo_id == "query":
#                             pos = query_servo_position(angle)
#                             if pos is None:
#                                 print("No response from Arduino for position query.")
#                             else:
#                                 if angle is None:
#                                     b_pos, s_pos, c_pos = pos.split(",")
#                                     print(f"Positions - Base: {b_pos}, Arm: {s_pos}, Claw: {c_pos}")
#                                 else:
#                                     print(f"Position of {angle}: {pos}")
#                         else:
#                             send_servo_command(servo_id, angle, speed)
#                         time.sleep(0.5)


import serial
import sys
import sounddevice as sd
import queue
from vosk import Model, KaldiRecognizer
import json
import re
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# SETTINGS
# =====================
ARDUINO_PORT = "COM8"
BAUD_RATE = 9600
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
DEFAULT_SPEED = 35  # degrees per step unless overridden
SIMILARITY_THRESHOLD = 0.6  # cosine similarity threshold for servo detection

# =====================
# NUMBER WORDS CONVERSION
# =====================
NUM_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
    'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14,
    'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19,
    'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70,
    'eighty':80, 'ninety':90, 'hundred':100
}

# Extra aliases for misheard words
SERVO_ALIASES = {
    "clock": "claw",
    "claude": "claw",
    "closer": "claw",
    "claus": "claw",
    "cloth": "claw"
}

def text2num(text):
    """
    Parse number words from text and return integer.
    Supports multi-word numbers like 'one hundred twenty three'.
    Returns None if no number found.
    """
    tokens = text.lower().split()
    total = 0
    current = 0
    found = False

    for token in tokens:
        if token not in NUM_WORDS:
            if found:
                break
            else:
                continue

        found = True
        val = NUM_WORDS[token]

        if val == 100:
            if current == 0:
                current = 1
            current *= 100
        else:
            current += val

    total += current

    if found:
        return total
    else:
        nums = re.findall(r'\b\d+\b', text)
        if nums:
            return int(nums[0])
        return None

# =====================
# INIT
# =====================
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
except serial.SerialException as e:
    print(f"Error connecting to Arduino: {e}")
    sys.exit(1)

q = queue.Queue()
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)
nlp = spacy.load("en_core_web_sm")

servo_names = ["base", "arm", "claw"]
servo_ids = {"base": "b", "arm": "s", "claw": "c"}

vectorizer = TfidfVectorizer().fit(servo_names)

def fuzzy_servo_match(word):
    word = word.lower()
    if word in SERVO_ALIASES:
        word = SERVO_ALIASES[word]
    word_vec = vectorizer.transform([word])
    sims = cosine_similarity(word_vec, vectorizer.transform(servo_names)).flatten()
    max_idx = sims.argmax()
    max_sim = sims[max_idx]
    if max_sim >= SIMILARITY_THRESHOLD:
        return servo_ids[servo_names[max_idx]]
    return None

def parse_command(text):
    text = text.lower().strip()

    # Special "home" command
    if text == "home":
        return ("home", None, None)

    doc = nlp(text)
    servo_id = None
    for token in doc:
        candidate = fuzzy_servo_match(token.text)
        if candidate:
            servo_id = candidate
            break

    query_phrases = ["what is", "current position", "position of", "where is", "show position", "get position", "gimme position", "position"]
    is_query = any(phrase in text for phrase in query_phrases)

    if is_query:
        if servo_id:
            return ("query", servo_id, None)
        else:
            return ("query", None, None)

    speed = DEFAULT_SPEED
    speed_match = re.search(r'speed\s+(\d+)', text)
    if speed_match:
        speed = int(speed_match.group(1))

    angle = text2num(text)
    if angle is None:
        angle = 90

    if servo_id is None:
        print(f"Could not detect servo in '{text}'")
        return None

    return (servo_id, angle, speed)

def send_servo_command(servo_id, angle, speed):
    cmd = f"{servo_id} {angle} {speed}\n"
    arduino.write(cmd.encode())
    print(f"Sent: {cmd.strip()}")

def home_position():
    send_servo_command("b", 90, DEFAULT_SPEED)  # base
    send_servo_command("s", 0, DEFAULT_SPEED)   # arm
    send_servo_command("c", 20, DEFAULT_SPEED)  # claw
    print("Moved to home position.")

def query_servo_position(servo_id=None):
    if servo_id is None:
        cmd = "getpos\n"
    else:
        cmd = f"getpos {servo_id}\n"
    arduino.write(cmd.encode())

    start_time = time.time()
    while True:
        if arduino.in_waiting:
            response = arduino.readline().decode().strip()
            if response:
                return response
        if time.time() - start_time > 2:
            return None

def interpret_commands(text):
    steps = re.split(r'\band then\b|\bthen\b|\band\b', text.lower())
    return [step.strip() for step in steps if step.strip()]

print("Listening for multi-step commands...")

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=lambda indata, frames, time_info, status: q.put(bytes(indata))):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result and result["text"].strip():
                heard = result["text"]
                print(f"Heard: {heard}")
                for step in interpret_commands(heard):
                    parsed = parse_command(step)
                    if parsed:
                        if parsed[0] == "home" or parsed[0] =="can you home":
                            home_position()
                        elif parsed[0] == "query":
                            pos = query_servo_position(parsed[1])
                            if pos is None:
                                print("No response from Arduino for position query.")
                            else:
                                if parsed[1] is None:
                                    b_pos, s_pos, c_pos = pos.split(",")
                                    print(f"Positions - Base: {b_pos}, Arm: {s_pos}, Claw: {c_pos}")
                                else:
                                    print(f"Position of {parsed[1]}: {pos}")
                        else:
                            servo_id, angle, speed = parsed
                            send_servo_command(servo_id, angle, speed)
                        time.sleep(0.5)
