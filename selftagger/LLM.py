
from openai import OpenAI
import httpx
from selftagger.conf import Config
import json

class LLM:
    REPEAT = "--- === REPEAT CALL === ---"
    def __init__(self, system):
        conf = Config()
        self.http_client = httpx.Client(trust_env=False)
        self.llm = OpenAI(base_url = conf.llmBaseUrl,
                          api_key = conf.llmApiKey,
                          project = conf.llmProject,
                          http_client = self.http_client)
        self.conf = conf
        self.messages = [{"role": "system", 
             "content": system }]

    def run(self, tools, functions):
        resp = self.llm.chat.completions.create(
            model = self.conf.llmModel,
            messages = self.messages,
            tools = tools,
            timeout=180.0)
        self.messages.append(resp.choices[0].message)
        message = resp.choices[0].message
        if not message.tool_calls:
            return message.content

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            if name not in functions:
                raise ValueError("No function " + name)
            result = functions[name](**args)
            self.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": str(result)
            })
        return self.REPEAT
    


    def add_user_message(self, message):
        self.messages.append({"role": "user",
                               "content": message })
        
