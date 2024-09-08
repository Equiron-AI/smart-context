import requests
import sseclient
import json


class LlamaCppBackend:
    def __init__(self, url, max_predict=1024):
        self.url = url
        self.max_predict = max_predict

    def get_request_object(self, request_tokens, stream, temp, top_p):
        return {"prompt": request_tokens,
                "stream": stream,
                "n_predict": self.max_predict,
                "temperature": temp,
                "repeat_last_n": 0,
                "repeat_penalty": 1.0,
                "top_k": -1,
                "top_p": top_p,
                "min_p": 0,
                "tfs_z": 1,
                "typical_p": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "stop": [self.stop_token, self.tokenizer.eos_token],
                "cache_prompt": True}

    def completion(self, request_tokens, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, False, temp, top_p)
        response = requests.post(self.url, json=request)
        response.raise_for_status()
        return response.json()["content"]

    async def stream_completion(self, request_tokens, callback, temp=0.5, top_p=0.5):
        request = self.get_request_object(request_tokens, True, temp, top_p)
        response = requests.post(self.url, json=request, stream=True, headers={'Accept': 'text/event-stream'})
        response.raise_for_status()
        stream = sseclient.SSEClient(response).events()
        text_resp = ""
        for event in stream:
            parsed_event = json.loads(event.data)
            if parsed_event["stop"]:
                break
            content = parsed_event["content"]
            text_resp += content
            await callback(content)
        return text_resp
