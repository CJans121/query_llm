#include "query_llm_behavior/query_llm_behavior.hpp"

namespace query_llm_behavior {

QueryLlm::QueryLlm(const std::string &name, const BT::NodeConfig &config)
    : BT::SyncActionNode(name, config), Node(name) {}

BT::PortsList QueryLlm::providedPorts() {
  return {
    BT::InputPort<std::string>("llm_model"),
    BT::InputPort<std::string>("llm_prompt"),
    BT::InputPort<std::shared_ptr<std::vector<spot_utils::StampedImage>>>("stamped_image_list"),
    BT::OutputPort<std::shared_ptr<std::string>>("llm_answer"),
    BT::OutputPort<std::shared_ptr<void>>("llm_parsed_answer")
  };
}

BT::NodeStatus QueryLlm::tick() {
  // Extract llm_model with fallback to default
  std::string llm_model;
  if (auto llm_model_exp = getInput<std::string>("llm_model"); llm_model_exp) {
    llm_model = llm_model_exp.value();
  } else {
    RCLCPP_WARN(this->get_logger(), "Input [llm_model] not specified. Using default: %s", default_llm_model_);
    llm_model = default_llm_model_;
  }

  // Extract llm_prompt
  const auto llm_prompt_exp = getInput<std::string>("llm_prompt");
  if (!llm_prompt_exp) {
    throw BT::RuntimeError("Missing or invalid input [llm_prompt]");
  }
  const std::string &llm_prompt = llm_prompt_exp.value();

  // Extract stamped_image_list
  using StampedImageVecPtr = std::shared_ptr<std::vector<spot_utils::StampedImage>>;
  const auto stamped_image_list_exp = getInput<StampedImageVecPtr>("stamped_image_list");
  if (!stamped_image_list_exp) {
    throw BT::RuntimeError("Missing or invalid input [stamped_image_list]");
  }

  const auto &stamped_image_list = *stamped_image_list_exp.value();

  // Call LLM
  const std::string answer = get_llm_answer_(llm_model, llm_prompt, stamped_image_list);
  RCLCPP_INFO(this->get_logger(), "LLM response:\n%s", answer.c_str());

  // Set output port
  setOutput("llm_answer", std::make_shared<std::string>(answer));

  // Check if the user has overridden parse_llm_answer()
  if (auto parsed = parse_llm_answer(answer); parsed) {
    setOutput("llm_parsed_answer", parsed);
  }
  return BT::NodeStatus::SUCCESS;
}

std::string QueryLlm::get_llm_answer_(
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

} // namespace query_llm_behavior

