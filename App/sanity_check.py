from os import path
import argparse
import importlib
import inspect
import sys

FAIL_COLOR = '\033[91m'
OK_COLOR = '\033[92m'
WARN_COLOR = '\033[93m'


def run_sanity_check(test_dir):

    print('Perform a sanity test to ensure code works properly.\n')
    print('Path to the file w/ test cases for  GET() and POST() methods')
    print('The path should be something like abc/def/test_xyz.py')
    filepath = input('> ')

    assert path.exists(filepath), f"File {filepath} does not exist."
    sys.path.append(path.dirname(filepath))

    module_name = path.splitext(path.basename(filepath))[0]
    module = importlib.import_module(module_name)

    test_function_names = list(
        filter(
            lambda x: inspect.isfunction(
                getattr(
                    module,
                    x)) and not x.startswith('__'),
            dir(module)))

    test_functions_for_get = list(
        filter(
            lambda x: inspect.getsource(
                getattr(
                    module,
                    x)).find('.get(') != -1,
            test_function_names))
    test_functions_for_post = list(
        filter(
            lambda x: inspect.getsource(
                getattr(
                    module,
                    x)).find('.post(') != -1,
            test_function_names))

    print("\n============= Sanity Check Report ===========")
    SANITY_TEST_PASSING = True
    WARNING_COUNT = 1

    # GET()
    TEST_FOR_GET_METHOD_RESPONSE_CODE = False
    TEST_FOR_GET_METHOD_RESPONSE_BODY = False
    if not test_functions_for_get:
        print(FAIL_COLOR + f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR + "No test cases detected for GET().")
        print(
            FAIL_COLOR +
            "\n Ensure a test case is written for the GET method.\
            Test both the status code as well as the contents\
            of the request object.\n")
        SANITY_TEST_PASSING = False

    else:
        for func in test_functions_for_get:
            source = inspect.getsource(getattr(module, func))
            if source.find('.status_code') != -1:
                TEST_FOR_GET_METHOD_RESPONSE_CODE = True
            if (source.find('.json') != -1) or (source.find(
                    'json.loads') != -1):
                TEST_FOR_GET_METHOD_RESPONSE_BODY = True

        if not TEST_FOR_GET_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                " GET() test isn't testing the response code.\n")

        if not TEST_FOR_GET_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                "GET() test isn't testing the CONTENTS of the response.\n")

    # POST()
    TEST_FOR_POST_METHOD_RESPONSE_CODE = False
    TEST_FOR_POST_METHOD_RESPONSE_BODY = False
    COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT = 0

    if not test_functions_for_post:
        print(FAIL_COLOR + f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR + "No test cases were detected for POST() .")
        print(
            FAIL_COLOR +
            "Must have two test cases for the POST() method." +
            "\nOne test case for EACH of the possible model inferences.\n")
        SANITY_TEST_PASSING = False
    else:
        if len(test_functions_for_post) == 1:
            print(f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                "Only one test case was detected for the POST() method.")
            print(
                FAIL_COLOR +
                "Must have two test cases for the POST() method." +
                "\nOne test case for EACH possible model inference.\n")
            SANITY_TEST_PASSING = False

        for func in test_functions_for_post:
            source = inspect.getsource(getattr(module, func))
            if source.find('.status_code') != -1:
                TEST_FOR_POST_METHOD_RESPONSE_CODE = True
            if (source.find('.json') != -1) or\
                    (source.find('json.loads') != -1):
                TEST_FOR_POST_METHOD_RESPONSE_BODY = True
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT += 1

        if not TEST_FOR_POST_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                "Test cases for POST() aren't testing the response code.\n")
        if not TEST_FOR_POST_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                "POST() tests aren't testing the contents of the response.\n")

        if len(
                test_functions_for_post) >= 2 and\
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT < 2:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR +
                "Need TWO test cases, one for each possible prediction.")

    SANITY_TEST_PASSING = SANITY_TEST_PASSING and\
        TEST_FOR_GET_METHOD_RESPONSE_CODE and \
        TEST_FOR_GET_METHOD_RESPONSE_BODY and \
        TEST_FOR_POST_METHOD_RESPONSE_CODE and \
        TEST_FOR_POST_METHOD_RESPONSE_BODY and \
        COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT >= 2

    if SANITY_TEST_PASSING:
        print(OK_COLOR + "Your test cases look good!")

    print(
        WARN_COLOR +
        "This is a heuristic based sanity testing.\
        Cannot guarantee code's correctness.")
    print(
        WARN_COLOR +
        "Sill check code directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_dir',
        metavar='test_dir',
        nargs='?',
        default='tests',
        help='Name of the directory that has test files.')
    args = parser.parse_args()
    run_sanity_check(args.test_dir)
