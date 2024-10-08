enum Topic {
    Prompting @description(#"
        Concerns how to structure plain text to ask the LLM to do something.
    "#)
    AI_Agents @description(#"
        LLMs that have access to tooling. They are given goals and can execute workflows autonomously.
    "#)
    RAG
    LLMs
    Cost
    BAML
    Other
}

class Question {
    question string
    answer string
    topic Topic @description(#"
        applies to question
    "#)
    blog BlogQ @description(#"
        how to pose this quesiton in a follow up blog post about the webinar
    "#)
}

class BlogQ {
    question string
    answer string
}

function ExtractQuestions(transcript: string) -> Question[] {
    client CustomGPT4o
    prompt #"
        Find all the questions users asksed related to BAML or building AI pipeleins.
        {{ ctx.output_format }}

        Before answering, list out all key questions that may apply to developers interested in learning about AI by topics.

        EXAMPLE
        - ...
        - ...
        ...

        [
            {
            .. // SCHEMA
            }
        ]

        {{ _.role('user') }}
        {{ transcript }}
    "#
}
