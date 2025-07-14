/**
 * @file query_llm_behavior.hpp
 * @author Janak Panthi (Crasun Jans)
 * @brief Behavior Tree node that queries a multimodal LLM using model name, prompt and optional images.
 *
 * Returns the LLM response as a string via an output port. A subclass may override
 * `parse_llm_answer()` to convert the raw string into a structured object, which is
 * then provided via a separate output port.
 */
#ifndef QUERY_LLM_BEHAVIOR_HPP
#define QUERY_LLM_BEHAVIOR_HPP

#include <behaviortree_cpp/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include <spot_utils/camera_client.hpp>
#include <string>
#include <memory>
#include <vector>
#include <sstream>

#include "ollama/ollama.hpp"
#include "nlohmann/json.hpp"

namespace query_llm_behavior {

/**
 * @brief Behavior Tree node that queries a vision-capable LLM using a textual prompt
 *        and an optional sequence of base64-encoded images.
 *
 * This node integrates into a Behavior Tree and performs a synchronous action.
 * It supports model selection, prompt injection, optional image-based context,
 * and allows subclasses to implement custom logic to parse the raw LLM response
 * into a structured object.
 */
template<typename ParsedType = void>
class QueryLlm : public BT::SyncActionNode, public rclcpp::Node {
public:
  /**
   * @brief Constructor for QueryLlm node.
   *
   * @param name Name of the Behavior Tree node.
   * @param config BT node configuration structure.
   */
  inline QueryLlm(const std::string &name, const BT::NodeConfig &config)
      : BT::SyncActionNode(name, config), Node(name) {}

  /**
   * @brief Defines all input/output ports for this node.
   *
   * @return BT::PortsList containing all the ports used by this node.
   * - Input:
   *   - `llm_model` (`std::string`, optional): Model identifier (e.g. "llava"). Defaults to `"llava"`.
   *   - `llm_prompt` (`std::string`, required): Prompt to be passed to the model.
   *   - `stamped_image_list` (`std::shared_ptr<std::vector<spot_utils::StampedImage>>`): Images with metadata.
   *
   * - Output:
   *   - `llm_answer` (`std::shared_ptr<std::string>`): Raw string response from the LLM.
   *   - `llm_parsed_answer` (`std::shared_ptr<ParsedType>`): Optionally parsed structured result.
   */
  inline static BT::PortsList providedPorts() {
    return {
      BT::InputPort<std::string>("llm_model"),
      BT::InputPort<std::string>("llm_prompt"),
      BT::InputPort<std::shared_ptr<std::vector<spot_utils::StampedImage>>>("stamped_image_list"),
      BT::OutputPort<std::shared_ptr<std::string>>("llm_answer"),
      BT::OutputPort<std::shared_ptr<ParsedType>>("llm_parsed_answer")
    };
  }

  /**
   * @brief Ticks the node once and performs the LLM query.
   *
   * This method retrieves inputs (model name, prompt, image list), calls the LLM API,
   * and sets output ports accordingly.
   *
   * @return NodeStatus::SUCCESS if query completed, throws otherwise.
   */
  inline BT::NodeStatus tick() override {
    std::string llm_model;
    if (auto llm_model_exp = getInput<std::string>("llm_model"); llm_model_exp) {
      llm_model = llm_model_exp.value();
    } else {
      RCLCPP_WARN(this->get_logger(), "Input [llm_model] not specified. Using default: %s", default_llm_model_);
      llm_model = default_llm_model_;
    }

    // Preload model
    this->load_model_(llm_model);

    const auto llm_prompt_exp = getInput<std::string>("llm_prompt");
    if (!llm_prompt_exp) {
      throw BT::RuntimeError("Missing or invalid input [llm_prompt]");
    }
    const std::string &llm_prompt = llm_prompt_exp.value();

    using StampedImageVecPtr = std::shared_ptr<std::vector<spot_utils::StampedImage>>;
    const auto stamped_image_list_exp = getInput<StampedImageVecPtr>("stamped_image_list");
    if (!stamped_image_list_exp) {
      throw BT::RuntimeError("Missing or invalid input [stamped_image_list]");
    }

    const auto &stamped_image_list = *stamped_image_list_exp.value();
    const std::string answer = get_llm_answer_(llm_model, llm_prompt, stamped_image_list);

    RCLCPP_INFO(this->get_logger(), "LLM response:\n%s", answer.c_str());
    setOutput("llm_answer", std::make_shared<std::string>(answer));

    if (auto parsed = parse_llm_answer(answer); parsed) {
      setOutput("llm_parsed_answer", parsed);
    }
    return BT::NodeStatus::SUCCESS;
  }

protected:
  /**
   * @brief Optionally override this in subclasses to extract structured data from the LLM output.
   *
   * If overridden and returns a non-null result, it will be published to the `llm_parsed_answer` port.
   *
   * @param raw_answer The raw string returned from the LLM.
   * @return A shared pointer to the parsed result (or nullptr if unused).
   */
  inline virtual std::shared_ptr<ParsedType> parse_llm_answer([[maybe_unused]] const std::string &raw_answer) {
    return nullptr;
  }

private:

  /**
   * @brief Ensures that the specified LLM model is available locally by checking,
   *        pulling, and loading as needed.
   *
   * This method first checks the list of locally available models to determine
   * whether the target model is already cached. If not found, it attempts to pull
   * it from the Ollama server. After pulling, it verifies the model can be loaded.
   *
   * @param llm_model Name of the LLM model (e.g., "llava" or "llama3:8b").
   * @throws std::runtime_error if model listing, pulling, or loading fails.
   */
  inline void load_model_(const std::string &llm_model) {
    // Check if the model is already listed (loaded) locally 
    try {
      const auto models = ollama::list_models();
      const auto found = std::find(models.begin(), models.end(), llm_model);
      if (found != models.end()) {
        RCLCPP_DEBUG(this->get_logger(), "Model [%s] is already available locally.", llm_model.c_str());
        return;
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to list local models: " + std::string(e.what()));
    }
  
    // Pull the model from the remote server if not found locally
    RCLCPP_INFO(this->get_logger(), "Model [%s] not found locally. Attempting to pull...", llm_model.c_str());
  
    try {
      if (!ollama::pull_model(llm_model)) {
        throw std::runtime_error(
            "Failed to pull model: " + llm_model + ". The model may not exist or the name may be incorrect.");
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Exception occurred while pulling model [" + llm_model + "]: " + std::string(e.what()));
    }
  
    // Load the model into memory after successful pull
    RCLCPP_INFO(this->get_logger(), "Model [%s] successfully pulled. Attempting to load...", llm_model.c_str());
  
    try {
      if (!ollama::load_model(llm_model)) {
        throw std::runtime_error("Model pulled but still failed to load: " + llm_model);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Exception occurred while loading model [" + llm_model + "]: " + std::string(e.what()));
    }
  
    RCLCPP_INFO(this->get_logger(), "Model [%s] successfully loaded.", llm_model.c_str());
  }
  
    
  /**
   * @brief Queries the LLM with the model, prompt, and associated images.
   *
   * This constructs a generation-style request and parses the JSON response.
   *
   * @param llm_model Model to be used (e.g., "llava").
   * @param llm_prompt User prompt with optional image references.
   * @param stamped_base64_img_list List of images in base64 format with IDs.
   * @return The raw text response from the LLM.
   */
  inline std::string get_llm_answer_(
      const std::string &llm_model,
      const std::string &llm_prompt,
      const std::vector<spot_utils::StampedImage> &stamped_base64_img_list) {

    std::vector<std::string> images;
    std::ostringstream img_order;

    for (size_t i = 0; i < stamped_base64_img_list.size(); ++i) {
      img_order << "[" << i + 1 << "] " << stamped_base64_img_list[i].id << "\n";
      images.push_back(stamped_base64_img_list[i].base64_image);
    }

    const std::string full_prompt = llm_prompt + "\nImages are provided in the following order:\n" + img_order.str();

    try {
      ollama::request req(ollama::message_type::generation);
      req["model"] = llm_model;
      req["prompt"] = full_prompt;
      req["max_tokens"] = max_tokens_;
      req["images"] = images;

      const ollama::response resp = ollama::generate(req);
      const auto jresp = resp.as_json();

      if (!jresp.contains("response") || !jresp["response"].is_string()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid or missing 'response' field from LLM.");
        return "";
      }

      return jresp["response"];
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "LLM call failed: %s. Is your model loaded?", e.what());
      return "";
    }
  }

  /// Default LLM model name (used when no model is specified via input port).
  static constexpr const char *default_llm_model_ = "llava";

  /// Maximum number of tokens allowed in the LLM response.
  const int max_tokens_ = 200;
};

} // namespace query_llm_behavior

#endif // QUERY_LLM_BEHAVIOR_HPP
