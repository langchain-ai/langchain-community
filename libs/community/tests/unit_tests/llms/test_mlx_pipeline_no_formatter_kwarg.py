from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.prompts import PromptTemplate

def test_mlx_pipeline_no_formatter_kwarg():
    """Mimicking example code listed on https://python.langchain.com/docs/integrations/llms/mlx_pipelines/

    To showcase issue with `formatter` keyword which is being passed to mlx-lm
    generate function
    """
    pipe = MLXPipeline.from_model_id(
         "mlx-community/quantized-gemma-2b-it"
     )

    assert pipe, "Model not loaded"

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    chain = prompt | pipe

    question = "What is electroencephalography?"

    answer = chain.invoke({"question": question})

    assert answer, "Answer was not generated"


    
