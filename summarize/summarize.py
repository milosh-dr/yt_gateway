# Update imports and venv

# Update the code after experimenting in Colab
import os

from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Get the OpenAI api key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_video_id(url):
    """Retrieves a video_id from a valid YouTube url.
    """
    splitted = url.split('watch?v=') 
    if len(splitted) != 2:
        return None
    
    return splitted[1]
    

def get_transcript(video_id):
    """Retrieves a transcript of a YouTube video"""
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id)
        lines = []
        for line in tr:
            lines.append(line['text'])

        tr = ' '.join(lines)
        return tr, None
    except Exception as err:
        print('Transcript not available for this video')
        return None, err


def split_document(document):
    """Splits document based on the length if needed"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 12000,
        chunk_overlap = 0
    )

    docs = text_splitter.create_documents([document])
    return docs


def gpt_single_doc(doc):

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    base_prompt = """Your job is to generate around 800 tokens long notes that a university student might create on a lecture.
    The text you are going to use is a neuroscience podcast transcript. Make sure you include only points coming directly from the text.
    Have the following principles in mind.
    1. Exclude any information on podcast's sponsors and ways of supporting it.
    2. Eliminate redunduncies if necessary
    3. Edit the notes to a neatly formated, easy to follow bullet-style (possibly nested) structure
    
    Text:"""

    try:
        response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {
                                'role': 'user',
                                'content': '\n'.join([base_prompt, doc])
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
    except Exception as err:
        print(err)
        return None, err
    
    summary = response.choices[0].message.content
    return summary, None


def gpt_multiple_docs(docs):

    client = OpenAI(api_key=OPENAI_API_KEY)

    responses = []

    base_prompt = """Your job is to generate around 800 tokens long possibly nested bullet-style notes that a university student might create on a lecture.
    The text you are going to use is a neuroscience podcast transcript. Make sure you include only points coming directly from the text.
    
    Text:"""

    for doc in docs:
        tmp_text = doc.page_content

        response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            'role': 'user',
                            'content': '\n'.join([base_prompt, tmp_text])
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        responses.append(response)
    
    text_for_gpt4 = ""

    for r in responses:
        text_for_gpt4 = "\n\n".join([text_for_gpt4, r.choices[0].message.content])
    
    text_for_gpt4 = text_for_gpt4.lstrip()

    # Make sure the lenght is ok, for now just print
    print(f'The length of final text for gpt4: {len(text_for_gpt4)} characters (approx. {len(text_for_gpt4)//4} tokens)')

    gpt4_prompt = """
    Given the following notes made from several chunks of the same podcast transcript create the final version with the following principles in mind.
    1. Exclude any information on podcast's sponsors and ways of supporting it.
    2. Eliminate redunduncies
    3. Edit the notes to a neatly formated, easy to follow bullet-style (possibly nested) structure

    Notes:
    """
    try:
        final_response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {
                                'role': 'user',
                                'content': '\n'.join([gpt4_prompt, text_for_gpt4])
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
    except Exception as err:
        print(err)
        return None, err
    
    summary = final_response.choices[0].message.content
    return summary, None


def get_summary(docs):

    if len(docs) == 1:
        summary = gpt_single_doc(docs[0])
        return summary, None

    summary = gpt_multiple_docs(docs)
    return summary, None    


def summarize(url):
    
    video_id = get_video_id(url)
    if not video_id:
        return None, "Provide the valid YouTube url"
    
    tr, err = get_transcript(video_id)
    if err:
        return None, err

    # Deal with large inputs
    docs = split_document(tr)

    summary = get_summary(docs)

    return summary, None


# For testing purposes
if __name__ == '__main__':
    summarize('')