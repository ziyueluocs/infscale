# ServeSpec

ServiceSpec model.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** |  | 
**model** | **str** |  | 
**num_failures** | **int** |  | 

## Example

```python
from infscale.openapi.models.serve_spec import ServeSpec

# TODO update the JSON string below
json = "{}"
# create an instance of ServeSpec from a JSON string
serve_spec_instance = ServeSpec.from_json(json)
# print the JSON string representation of the object
print ServeSpec.to_json()

# convert the object into a dict
serve_spec_dict = serve_spec_instance.to_dict()
# create an instance of ServeSpec from a dict
serve_spec_form_dict = serve_spec.from_dict(serve_spec_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


