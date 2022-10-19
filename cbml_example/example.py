sys.path.append('../') # in this case, the arstpy library is one folder above the cbml_example folder
import json,sys,signal
from collections import OrderedDict
from datetime import datetime
from arstpy import cbml
def signal_handler(sig, frame):
    print('Output: Have a nice day!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def get_fresh_string_replacements():
    return OrderedDict([
        ("$MACHINE_NAME","Arisha"),
        ("$MACHINE_VERSION","1.0"),
        ("$MACHINE_AGE","21"),
        ("$MACHINE_BIRTHDATE","June 05, 2001"),
        ("$MACHINE_AUTHOR","Rizaldy Aristyo"),
        ("$OFFICIAL_WEBSITE","https://github.com/ARSTCreations"),
        ("$CURRENT_TIME",str(datetime.now().strftime("%I:%M:%S%p"))),
        ("$CURRENT_DATE",str(datetime.now().strftime("%B, %d, %Y"))),
        ("$CURRENT_DAY",str(datetime.now().strftime("%A"))),
    ])

def start():
    # cbml.load("corpus.json", verbosity=0)
    cbml.train_and_load("corpus.json",verbosity=2)
    # Regular Response
    print(cbml.respond("Hello"))
    # Response with String Replacement
    print(cbml.respond("What's your name?", replacement_ordereddict=get_fresh_string_replacements()))
    # Response with Custom Unrecognized Response, Ambiguous Response, and String Replacement
    print(cbml.respond("Bye!", "I Don't Understand", "...?", replacement_ordereddict=get_fresh_string_replacements()))
    # JSON Response
    print("JSON:",json.loads(cbml.respond_json("Hello", replacement_ordereddict=get_fresh_string_replacements())))
    # On an infinite loop
    while True: print('Output:', cbml.respond(input("Input: "), replacement_ordereddict=get_fresh_string_replacements()))

start()