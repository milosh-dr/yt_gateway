import json
from flask import Flask, request

from auth import validate
from auth_svc import access

from summarize import summarize

server = Flask(__name__)

@server.route('/login', methods=['POST'])
def login():
    token, err = access.login(request)
    if not err:
        return token
    else:
        return err

@server.route('/summarize', methods=['GET'])
def summarize():
    access, err = validate.token(request)
    if err:
        return err

    access = json.loads(access)

    if not access['admin']:
        return "Not authorized", 401
    
    url = request.args.get['url']
    
    if not url:
        return "Provide the url to the YouTube video", 400
    
    summary, err = summarize.summarize(url)

    if err:
        return err
    
    return summary


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8080)