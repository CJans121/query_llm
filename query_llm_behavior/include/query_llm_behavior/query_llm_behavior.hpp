/**
 * @file query_llm.hpp
 * @author Janak Panthi (Crasun Jans)
 * @brief Behavior Tree node that queries a multimodal LLM using model name, prompt and optional images.
 *
 * Returns the LLM response as a string via an output port. A subclass may override
 * `parse_llm_answer()` to convert the raw string into a structured object, which is
 * then provided via a separate output port.
 */

#ifndef QUERY_LLM_HPP
#define QUERY_LLM_HPP

#include <behaviortree_cpp/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include <spot_utils/camera_client.hpp>
#include <string>
#include <memory>
#include <vector>

#include "ollama/ollama.hpp"
#include "nlohmann/json.hpp"

namespace query_llm {

/**
 * @brief Behavior Tree node that queries a vision-capable LLM using a textual prompt
 *        and an optional sequence of base64-encoded images.
 *
 * This node integrates into a Behavior Tree and performs a synchronous action.
 * It supports model selection, prompt injection, optional image-based context,
 * and allows subclasses to implement custom logic to parse the raw LLM response
 * into a structured object.
 */
class QueryLlm : public BT::SyncActionNode, public rclcpp::Node {
public:
  /**
   * @brief Constructor for QueryLlm node.
   *
   * @param name Name of the Behavior Tree node.
   * @param config BT node configuration structure.
   */
  QueryLlm(const std::string &name, const BT::NodeConfig &config);

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
   *   - `llm_parsed_answer` (`std::shared_ptr<void>`): Optionally parsed structured result.
   *
   */
  static BT::PortsList providedPorts();

  /**
   * @brief Ticks the node once and performs the LLM query.
   *
   * This method retrieves inputs (model name, prompt, image list), calls the LLM API,
   * and sets output ports accordingly.
   *
   * @return NodeStatus::SUCCESS if query completed, throws otherwise.
   */
  BT::NodeStatus tick() override;

protected:
  /**
   * @brief Optionally override this in subclasses to extract structured data from the LLM output.
   *
   * If overridden and returns a non-null result, it will be published to the `llm_parsed_answer` port.
   *
   * @param raw_answer The raw string returned from the LLM.
   * @return A shared pointer to the parsed result (or nullptr if unused).
   */
  virtual std::shared_ptr<void> parse_llm_answer([[maybe_unused]] const std::string &raw_answer) {
    return nullptr;
  }

private:
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
  std::string get_llm_answer_(
      const std::string &llm_model,
      const std::string &llm_prompt,
      const std::vector<spot_utils::StampedImage> &stamped_base64_img_list);

  /// Default LLM model name (used when no model is specified via input port).
  static constexpr const char *default_llm_model_ = "llava";

  /// Maximum number of tokens allowed in the LLM response.
  const int max_tokens_ = 200;
};

} // namespace query_llm

#endif // QUERY_LLM_HPP

