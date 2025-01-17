###############################################################################
#
#  Welcome to Baml! To use this generated code, please run the following:
#
#  $ pip install baml
#
###############################################################################

# This file was generated by BAML: please do not edit it. Instead, edit the
# BAML files and re-generate this code.
#
# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off
import baml_py
from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Union


class Topic(str, Enum):
    
    Prompting = "Prompting"
    AI_Agents = "AI_Agents"
    RAG = "RAG"
    LLMs = "LLMs"
    Cost = "Cost"
    BAML = "BAML"
    Other = "Other"

class BlogQ(BaseModel):
    
    
    question: str
    answer: str

class Question(BaseModel):
    
    
    question: str
    answer: str
    topic: "Topic"
    blog: "BlogQ"

class Resume(BaseModel):
    
    
    name: str
    email: str
    experience: List[str]
    skills: List[str]
