from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()


def main():
    information = """ Elon Reeve Musk (/ˈiːlɒn mʌsk/ Pretoria , June 28, 1971) is a South African -born American entrepreneur , investor , conservative political activist, [ 3 ] [ 4 ] and business magnate . [ note 1 ] He is the founder, CEO, and chief engineer of SpaceX ; an angel investor , managing director , and product architect of Tesla, Inc .; the founder of The Boring Company ; and a co-founder of Neuralink and OpenAI . [ note 2 ] He is also the chief technology officer of X Corp. [ 5 ] Between January and May 2025, he served as the de facto administrator of the White House Department of Government Efficiency under Donald Trump's second presidency . [ 6 ] [ 7 ] As of June 2026, Musk is the world's richest person according to Forbes . Following SpaceX 's initial public offering , his net worth exceeded US$1.1 trillion, making him the first and only billionaire in US dollar terms in history. [ 8 ] On June 16, 2026, his net worth reached a record high of US$1.4 trillion. [ 9 ]

Musk was born and raised in a wealthy family in Pretoria , South Africa . His mother is Canadian and his father is a white South African . He briefly studied at the University of Pretoria before moving to Canada at age 17. He enrolled at Queen's University and transferred to the University of Pennsylvania two years later, where he graduated with a degree in Economics and Physics. In 1995, he moved to California to attend Stanford University , but instead decided to pursue an entrepreneurial career, co-founding the web software company Zip2 with his brother Kimbal . Zip2 was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded the online bank X.com , which merged with Confinity in 2000 to form PayPal . The company was bought by eBay in 2002 for $1.5 billion.
"""

    summary_template = """given the information {information} about a person, I want you to create:
1. A short summary
2. Two interesting facts about them
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(model="gpt-oss:20b", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
