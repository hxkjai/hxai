import json
import threading

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


# https://github.com/volcengine/volc-sdk-python
class SAMIService(Service):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(SAMIService, "_instance"):
            with SAMIService._instance_lock:
                if not hasattr(SAMIService, "_instance"):
                    SAMIService._instance = object.__new__(cls)
        return SAMIService._instance

    def __init__(self):
        self.service_info = SAMIService.get_service_info()
        self.api_info = SAMIService.get_api_info()
        super(SAMIService, self).__init__(self.service_info, self.api_info)

    @staticmethod
    def get_service_info():
        api_url = 'open.volcengineapi.com'
        service_info = ServiceInfo(api_url, {},
                                   Credentials('', '', 'sami', 'cn-north-1'), 10, 10)
        return service_info

    @staticmethod
    def get_api_info():
        api_info = {
            "GetToken": ApiInfo("POST", "/", {"Action": "GetToken", "Version": "2021-07-27"}, {}, {}),
        }
        return api_info

    def common_json_handler(self, api, body):
        params = dict()
        try:
            body = json.dumps(body)
            res = self.json(api, params, body)
            res_json = json.loads(res)
            return res_json
        except Exception as e:
            res = str(e)
            try:
                res_json = json.loads(res)
                return res_json
            except: # noqa
                raise Exception(str(e))
            


    
