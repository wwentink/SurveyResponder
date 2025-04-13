import pytest
import os
import pandas as pd
from .. import SurveyResponder, load_questions, load_persona_file, generate_persona_from_file
from unittest.mock import patch

def test_surveyresponder_initialization(sample_questions_file, sample_persona_file):
    """Test SurveyResponder initialization with default parameters."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file
    )
    assert isinstance(responder, SurveyResponder)     # Verify object instantiation
    assert len(responder.questions) == 3              # Verify expected question count from fixture
    assert isinstance(responder.persona_dict, dict)   # Verify dictionary structure from fixture
    assert len(responder.response_options) == 5       # Verify default Likert scale length

def test_custom_response_options(sample_questions_file, sample_persona_file, custom_response_options):
    """Test SurveyResponder initialization with custom response options."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file,
        response_options=custom_response_options
    )
    assert responder.response_options == custom_response_options
    assert len(responder.response_options) == 5       # Verify default Likert scale length

def test_load_questions(sample_questions_file):
    """Test loading questions from file."""
    questions = load_questions(sample_questions_file)
    assert len(questions) == 3                         # Verify expected question count
    assert all(isinstance(q, str) for q in questions)  
    assert questions[0].startswith("Question 1")       # Verify question text from fixture

def test_load_persona_file(sample_persona_file):
    """Test loading persona file."""
    persona_dict = load_persona_file(sample_persona_file)
    assert isinstance(persona_dict, dict)              # Verify dictionary structure from fixture
    assert "age" in persona_dict                       # Verify presence of 'age' attribute
    assert "education" in persona_dict                 # Verify presence of 'education' attribute
    assert "occupation" in persona_dict                # Verify presence of 'occupation' attribute
    assert len(persona_dict["age"]) == 2               # Verify 'age' attribute value length

def test_generate_persona(sample_persona_file):
    """Test persona generation."""
    persona_dict = load_persona_file(sample_persona_file)
    traits, descriptions = generate_persona_from_file(persona_dict)
    
    assert isinstance(traits, dict)                   # Verify persona trait structure
    assert isinstance(descriptions, list)             # Verify persona description format
    assert len(traits) == len(persona_dict)           # Verify all traits were processed
    assert all(isinstance(d, str) for d in descriptions)  # Verify description string format

def test_example_prompt(sample_questions_file, sample_persona_file):
    """Test example prompt generation."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file
    )
    prompt = responder.example_prompt()
    assert isinstance(prompt, str)                    # Verify prompt is a string
    assert "Question" in prompt                       # Verify prompt contains question text
    assert all(
        opt.lower() in prompt.lower() 
        for opt in responder.response_options)        # Verify response options included

def test_example_persona(sample_questions_file, sample_persona_file):
    """Test example persona generation."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file
    )
    # Test single persona
    single_persona = responder.example_persona()
    assert isinstance(single_persona, str)            # Veryfiy persona is string
    
    # Test multiple personas
    multi_personas = responder.example_persona(npersonas=3)
    assert isinstance(multi_personas, list)           # Verify persona is a list
    assert len(multi_personas) == 3                   # Verify leng from fixture
    assert all(
        isinstance(p, str) for p in multi_personas)   # Veryfiy items in list are str

def test_get_settings(sample_questions_file, sample_persona_file):
    """Test getting settings."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file
    )
    settings = responder.get_settings()
    assert isinstance(settings, dict)
    assert settings["questions_path"] == sample_questions_file
    assert settings["persona_path"] == sample_persona_file
    assert settings["num_questions"] == 3

def test_run_with_mock_ollama(sample_questions_file, sample_persona_file, mock_ollama_response, ollama_available):
    """Test running the responder with mocked Ollama responses."""
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file,
        num_responses=2
    )
    df = responder.run()
    assert isinstance(df, pd.DataFrame)               # Verify DataFrame is returned
    assert len(df) == 2                               # Verify expected number of responses
    assert "resid" in df.columns                      # Verify response id column exists
    assert "model" in df.columns                      # Verify model name column exists
    assert "Q1" in df.columns                         # Verify question columns exist
    assert "Q2" in df.columns                         # Verify question columns exist
    assert "Q3" in df.columns                         # Verify question columns exist

def test_run_write_with_mock_ollama(sample_questions_file, sample_persona_file, mock_ollama_response, ollama_available, tmp_path):
    """Test running the responder and writing results to file."""
    output_file = os.path.join(tmp_path, "test_responses.csv")
    responder = SurveyResponder(
        questions_path=sample_questions_file,
        persona_path=sample_persona_file,
        num_responses=2
    )
    df = responder.run_write(output_file)
    
    assert isinstance(df, pd.DataFrame)              # Verify DataFrame is returned
    assert len(df) == 2                              # Verify expected number of responses
    
    assert os.path.exists(output_file)               # Verify output files were created
    assert os.path.exists(output_file.replace(".csv", "_params.json"))
    
    df_loaded = pd.read_csv(output_file)
    assert len(df_loaded) == 2                       # Verify CSV contains expected rows
    assert all(f"Q{i}" in df_loaded.columns          # Verify all question columns exist
               for i in range(1, 4))

def test_error_handling_invalid_files():
    """Test error handling for invalid files."""
    with pytest.raises(FileNotFoundError):            # Verify proper error handling
        SurveyResponder(
            questions_path="nonexistent_questions.txt",
            persona_path="nonexistent_persona.json"
        )
