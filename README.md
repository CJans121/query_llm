# query_llm
Houses a Behavior Tree node that queries a desired multimodal LLM (from Ollama or OpenAI) using a prompt and optional images. Puts the LLM's answer into a blackboard variable. Also allows for child classes to implement custom parsing of the LLM answer.

## Dependencies
- ROS humble
- Behaviortree cpp
- ollama : `curl -fsSL https://ollama.com/install.sh | sh`
- Pull desired ollama model as: `ollama pull <model_name>`

## Usage
### CMakeLists.txt addition
```cmake
find_package(query_llm_behavior REQUIRED)
```

### Using base class as is:
##### BT node registration
```cpp
#include <query_llm_behavior/query_llm_behavior.hpp>
factory.registerNodeType<query_llm_behavior::QueryLlm>("QueryLlm");
```
##### BT XML usage
```xml
<QueryLlm llm_model='gpt-4o' llm_prompt="Describe these images." image_list="{image_list}" llm_answer="{llm_answer}"/>
```

### Or with a child class, say `QueryVlm`
The child class implements the virtual function `parse_llm_answer` to parse the raw llm answer into a desired format.
#### BT node registration 
```cpp
// This is just an example. Adjust based on your implementation of the child class.
#include <query_vlm_behavior/query_vlm_behavior.hpp>
factory.registerNodeType<query_vlm_behavior::QueryVlm>("QueryVlm");
```
##### BT XML usage
```xml
<!-- Node name based on the above registration example -->
<QueryVlm llm_model='gemma3:27b-it-qat' llm_prompt="{llm_prompt}" image_list="{image_list}" llm_parsed_answer="{llm_parsed_answer}"/>
```

