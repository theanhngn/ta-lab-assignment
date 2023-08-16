"""
Unit tests for ta_asgn_mna.py and evo_hw_mna.py
"""
import ta_asgn_mna as tapy
from evo_hw import Evo
import pytest
import pandas as pd
import numpy as np


# read in + create test files
@pytest.fixture()
def test1():
    test1 = pd.read_csv('test1.csv', header=None)

    return test1


@pytest.fixture()
def test2():
    test2 = pd.read_csv('test2.csv', header=None)

    return test2


@pytest.fixture()
def test3():
    test3 = pd.read_csv('test3.csv', header=None)

    return test3


@pytest.fixture()
def test4():
    test4 = pd.DataFrame(np.ones((43, 17)), dtype=int)  # all 1s

    return test4


@pytest.fixture()
def test5():
    test5 = pd.DataFrame(np.zeros((43, 17)), dtype=int)  # all 0s

    return test5


# list of solutions
test_solutions = [test1, test2, test3]


# Tests for Objective Functions

def test_overallocation_score(test1, test2, test3, test4, test5):
    assert tapy.overallocation_score_func(test1) == 37, "Incorrect overallocation score for test 1 file"
    assert tapy.overallocation_score_func(test2) == 41, "Incorrect overallocation score for test 2 file"
    assert tapy.overallocation_score_func(test3) == 23, "Incorrect overallocation score for test 3 file"


def test_conflict_score(test1, test2, test3, test4, test5):
    assert tapy.conflict_score_func(test1) == 8, "Incorrect conflict score for test 1 file"
    assert tapy.conflict_score_func(test2) == 5, "Incorrect conflict score for test 2 file"
    assert tapy.conflict_score_func(test3) == 2, "Incorrect conflict score for test 3 file"


def test_undersupport_score(test1, test2, test3, test4, test5):
    assert tapy.undersupport_score_func(test1) == 1, "Incorrect under support score for test 1 file"
    assert tapy.undersupport_score_func(test2) == 0, "Incorrect under support score for test 2 file"
    assert tapy.undersupport_score_func(test3) == 7, "Incorrect under support score for test 3 file"


def test_unwilling_score(test1, test2, test3, test4, test5):
    assert tapy.willingness_score_func_U(test1) == 53, "Incorrect unwillingness score for test 1 file"
    assert tapy.willingness_score_func_U(test2) == 58, "Incorrect unwillingness score for test 2 file"
    assert tapy.willingness_score_func_U(test3) == 43, "Incorrect unwillingness score for test 3 file"


def test_unpreferred_score(test1, test2, test3, test4, test5):
    assert tapy.willingness_score_func_W(test1) == 15, "Incorrect unpreferred score for test 1 file"
    assert tapy.willingness_score_func_W(test2) == 19, "Incorrect unpreferred score for test 1 file"
    assert tapy.willingness_score_func_W(test3) == 10, "Incorrect unpreferred score for test 1 file"


# Test Evo framework constructor
def test_constructor():
    e = Evo()
    assert isinstance(e, Evo), "Did not construct a Evo instance"


# Test Agents
@pytest.fixture
def test_agent1():
    swapped_solution = tapy.random_swapper_col(test_solutions)
    return swapped_solution


def test_agent_df(test_agent1):
    assert test_agent1.shape[0] == 43, "The shape of the dataframe has changed"
    assert test_agent1.shape[1] == 17, "The shape of the dataframe has changed"


@pytest.fixture()
def test_agent2():
    swapped_solution = tapy.random_swapper_value(test_solutions)
    return swapped_solution


def test_agent_df2(test_agent2):
    assert test_agent2.shape[0] == 43, "The shape of the dataframe has changed"
    assert test_agent2.shape[1] == 17, "The shape of the dataframe has changed"


@pytest.fixture()
def test_agent3():
    swapped_solution = tapy.random_swapper_row_col(test_solutions)
    return swapped_solution


def test_agent_df3(test_agent3):
    assert test_agent3.shape[0] == 43, "The shape of the dataframe has changed"
    assert test_agent3.shape[1] == 17, "The shape of the dataframe has changed"


@pytest.fixture()
def test_agent4():
    swapped_solution = tapy.random_swapper_row(test_solutions)
    return swapped_solution


def test_agent_df4(test_agent4):
    assert test_agent4.shape[0] == 43, "The shape of the dataframe has changed"
    assert test_agent4.shape[1] == 17, "The shape of the dataframe has changed"
