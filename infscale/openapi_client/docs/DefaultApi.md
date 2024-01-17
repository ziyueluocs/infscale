# openapi_client.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**serve_models_post**](DefaultApi.md#serve_models_post) | **POST** /models | Serve


# **serve_models_post**
> Response serve_models_post(var_self, serve_spec)

Serve

Serve a model.

### Example


```python
import time
import os
import openapi_client
from openapi_client.models.response import Response
from openapi_client.models.serve_spec import ServeSpec
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    var_self = None # object | 
    serve_spec = openapi_client.ServeSpec() # ServeSpec | 

    try:
        # Serve
        api_response = api_instance.serve_models_post(var_self, serve_spec)
        print("The response of DefaultApi->serve_models_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->serve_models_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **var_self** | [**object**](.md)|  | 
 **serve_spec** | [**ServeSpec**](ServeSpec.md)|  | 

### Return type

[**Response**](Response.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

