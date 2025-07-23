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

// Cpp headers
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <future>
#include <chrono>

// Third-party headers
#include <behaviortree_cpp/action_node.h>
#include <opencv2/opencv.hpp>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>

// This packages headers
#include "ollama/ollama.hpp"
#include "nlohmann/json.hpp"
#include "base64/base64.hpp"

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
   *   - `image_list` (`std::shared_ptr<std::vector<sensor_msgs::msg::Image>>`): Images with metadata.
   *
   * - Output:
   *   - `llm_answer` (`std::shared_ptr<std::string>`): Raw string response from the LLM.
   *   - `llm_parsed_answer` (`std::shared_ptr<ParsedType>`): Optionally parsed structured result.
   */
  inline static BT::PortsList providedPorts() {
    return {
      BT::InputPort<std::string>("llm_model"),
      BT::InputPort<std::string>("llm_prompt"),
      BT::InputPort<std::shared_ptr<std::vector<sensor_msgs::msg::Image>>>("image_list"),
      BT::OutputPort<std::shared_ptr<std::string>>("llm_answer"),
      BT::OutputPort<std::shared_ptr<ParsedType>>("llm_parsed_answer")
    };
  }


  inline BT::NodeStatus onStart() override{

    // Get llm model
    std::string llm_model;
    if (auto llm_model_exp = getInput<std::string>("llm_model"); llm_model_exp) {
      llm_model = llm_model_exp.value();
    } else {
      RCLCPP_WARN(this->get_logger(), "Input [llm_model] not specified. Using default: %s", default_llm_model_);
      llm_model = default_llm_model_;
    }

		
    // Preload model with timeout using scoped future
    {
      // launch load_model_() in a background thread
      auto loader = std::async(std::launch::async, [this, llm_model]() {
        this->load_model_(llm_model);
      });

      // wait up to preload_timeout_
      if (loader.wait_for(preload_timeout_) 
          == std::future_status::timeout)
      {
        throw BT::RuntimeError("Loading LLM model timed out after " + std::to_string(
          std::chrono::duration_cast<std::chrono::minutes>(preload_timeout_).count()) + " minutes");
      }
      // rethrow any exception from load_model_
      loader.get();
    }

    // Get llm prompt
    const auto llm_prompt_exp = getInput<std::string>("llm_prompt");
    if (!llm_prompt_exp) {
      throw BT::RuntimeError("Missing or invalid input [llm_prompt]");
    }
    const std::string &llm_prompt = llm_prompt_exp.value();

    // Get image list
    using ImageVecPtr = std::shared_ptr<std::vector<sensor_msgs::msg::Image>>;
    const auto image_list_exp = getInput<ImageVecPtr>("image_list");
    if (!image_list_exp) {
      throw BT::RuntimeError("Missing or invalid input [image_list]");
    }

    const ImageVecPtr &image_list = image_list_exp.value();

    return BT::NodeStatus::RUNNING;
  
  }
	
  /**
   * @brief Ticks the node once and performs the LLM query.
   *
   * This method retrieves inputs (model name, prompt, image list), calls the LLM API,
   * and sets output ports accordingly.
   *
   * @return NodeStatus::SUCCESS if query completed, throws otherwise.
   */
  inline BT::NodeStatus onRunning(){

  std::string answer;

    // Run llm inference with timeout using scoped future
    { 
      // launch get_llm_answer_() in a background thread
      auto llm_future = std::async(
        std::launch::async,
        [this]() {
          return get_llm_answer_(llm_model, llm_prompt, image_list);
        }
      );

      // Wait for the LLM response up to llm_response_timeout_
      if (llm_future.wait_for(llm_response_timeout_)
          == std::future_status::timeout)
      {
        throw BT::RuntimeError(
          "LLM inference timed out after " +
          std::to_string(timeout_secs) +
          " seconds"
        );
      }
        // wait up to inference_timeout_
        if (llm_future.wait_for(inference_timeout_) 
            == std::future_status::timeout)
        {
          throw BT::RuntimeError("LLM inference timed out after " + std::to_string(
            std::chrono::duration_cast<std::chrono::seconds>(inference_timeout_).count()) + " seconds");
        }

      // Retrieve result (or rethrow any exception from get_llm_answer_)
      answer = llm_future.get();
    } 

    // Set the output
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
        RCLCPP_WARN(this->get_logger(), "Model [%s] is already available locally.", llm_model.c_str());
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
   * @param img_msg_list List of images as vector of sensor_msgs::msg::Image
   * @return The raw text response from the LLM.
   */
   inline std::string get_llm_answer_(
       const std::string &llm_model,
       const std::string &llm_prompt,
       const std::shared_ptr<std::vector<sensor_msgs::msg::Image>> &img_msg_list)
   {
     // Validate input
     if (!img_msg_list || img_msg_list->empty()) {
       RCLCPP_ERROR(this->get_logger(),
                    "Image list is null or empty; aborting LLM call.");
       return "";
     }
   
     // 1) Build the stitching‑order note
     std::ostringstream img_order;
     img_order << "Images have been stitched side‑by‑side in the following order:\n";
     for (size_t i = 0; i < img_msg_list->size(); ++i) {
       const auto &img = img_msg_list->at(i);
       img_order << "[" << (i + 1) << "] " << img.header.frame_id << "\n";
     }
   
     const std::string full_prompt =
         llm_prompt + "\n" + img_order.str();
   
     // Stitch into one panorama and encode
     const cv::Mat pano = stitch_images_horizontally_(*img_msg_list);
     if (pano.empty()) {
       RCLCPP_ERROR(this->get_logger(),
                    "Stitched image is empty; aborting LLM call.");
       return "";
     }
   
     std::string stitched_b64;
     try {
       stitched_b64 = convert_mat_to_base64_(pano, img_format_);
     } catch (const std::exception &e) {
       RCLCPP_ERROR(this->get_logger(),
                    "Failed to base64‑encode stitched image: %s", e.what());
       return "";
     }
   
     // Optional: save for debugging
     try {
       save_image_to_file(stitched_b64, "stitched_base64_image.jpg");
     } catch (const std::exception &e) {
       RCLCPP_WARN(this->get_logger(),
                   "Could not save debug image: %s", e.what());
     }
   
     // Build & send the request
     try {
       ollama::request req(ollama::message_type::generation);
       req["model"]      = llm_model;
       req["prompt"]     = full_prompt;
       req["max_tokens"] = max_tokens_;
       req["image"]      = stitched_b64;
   
       const ollama::response resp = ollama::generate(req);
       const auto jresp = resp.as_json();
   
       if (!jresp.contains("response") || !jresp["response"].is_string()) {
         RCLCPP_ERROR(this->get_logger(),
                      "Invalid or missing 'response' field from LLM.");
         return "";
       }
       return jresp["response"].get<std::string>();
   
     } catch (const std::exception &e) {
       RCLCPP_ERROR(this->get_logger(),
                    "LLM call failed: %s. Is your model loaded?", e.what());
       return "";
     }
   }

   /**
    * @brief Stitch multiple ROS images side-by-side into a single OpenCV matrix.
    *
    * This function converts each input @p sensor_msgs::msg::Image to an OpenCV
    * BGR8 cv::Mat using cv_bridge, performs type and validity checks, and pads
    * the images vertically to ensure consistent height before concatenating them
    * horizontally with OpenCV's @c cv::hconcat.
    *
    * If any image is invalid (empty, conversion failure, or mismatched type),
    * it is skipped with a warning including its @c frame_id. If no valid images
    * remain after filtering, an empty @c cv::Mat is returned.
    *
    * @param[in] ros_imgs
    *   A vector of ROS Image messages to be stitched. Each image is converted
    *   via cv_bridge::toCvCopy(..., sensor_msgs::image_encodings::BGR8).
    *
    * @return cv::Mat
    *   A single horizontally-stitched image. Returns an empty @c cv::Mat if
    *   @p ros_imgs is empty or none of the images are valid.
    *
    * @note
    *   - All images must have the same type; otherwise, mismatched ones are skipped.
    *   - Images with smaller height are zero-padded (black) at the bottom to match the tallest one.
    *   - Logs warnings or errors with the descriptive @c frame_id of each image for traceability.
    */
   
    inline cv::Mat stitch_images_horizontally_(
       const std::vector<sensor_msgs::msg::Image> &ros_imgs)
   {
     if (ros_imgs.empty()) {
       RCLCPP_WARN(this->get_logger(), "No images to stitch; returning empty Mat");
       return {};
     }
   
     std::vector<cv::Mat> mats;
     mats.reserve(ros_imgs.size());
   
     int max_rows = 0;
     int ref_type = -1;
   
     for (const auto &ros_img : ros_imgs) {
       const std::string &frame_id = ros_img.header.frame_id;
   
       try {
         auto cv_ptr = cv_bridge::toCvCopy(ros_img, sensor_msgs::image_encodings::BGR8);
         const cv::Mat &img = cv_ptr->image;
   
         if (img.empty()) {
           RCLCPP_WARN(this->get_logger(), "Image [%s] is empty; skipping.", frame_id.c_str());
           continue;
         }
   
         if (ref_type == -1) {
           ref_type = img.type();
         }
   
         if (img.type() != ref_type) {
           RCLCPP_WARN(this->get_logger(),
                       "Image [%s] has mismatched type. Expected type %d but got %d. Skipping.",
                       frame_id.c_str(), ref_type, img.type());
           continue;
         }
   
         max_rows = std::max(max_rows, img.rows);
         mats.push_back(img);
       } catch (const std::exception &e) {
         RCLCPP_ERROR(this->get_logger(),
                      "Image [%s] could not be converted: %s", frame_id.c_str(), e.what());
       }
     }
   
     if (mats.empty()) {
       RCLCPP_ERROR(this->get_logger(), "No valid images to stitch.");
       return {};
     }
   
     // Pad all images to match max height
     for (auto &img : mats) {
       if (img.rows < max_rows) {
         int pad = max_rows - img.rows;
         cv::copyMakeBorder(img, img, 0, pad, 0, 0,
                            cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
       }
     }
   
     cv::Mat stitched;
     try {
       cv::hconcat(mats, stitched);
     } catch (const std::exception &e) {
       RCLCPP_ERROR(this->get_logger(), "hconcat failed: %s", e.what());
       return {};
     }
   
     return stitched;
   }


  /**
   * @brief Converts a cv::Mat image to a base64-encoded string using the specified image format.
   *
   * This function first encodes the input image into the desired format (e.g., JPEG or PNG),
   * and then converts the resulting binary data into a base64-encoded string.
   *
   * @param input The OpenCV image (cv::Mat) to encode.
   * @param img_format_ Image format to use for encoding (e.g., "jpeg", "png").
   * @return Base64-encoded string of the compressed image.
   *
   * @throws std::runtime_error if encoding fails or the base64 string is empty.
   */
  inline std::string convert_mat_to_base64_(const cv::Mat &input,
                                       const std::string &img_format_) {
    // Encode the cv::Mat to a specified image format (e.g., JPEG, PNG)
    std::vector<unsigned char> buffer;
    std::vector<int> params;
  
    // Set encoding parameters for quality, if needed (e.g., JPEG quality)
    if (img_format_ == "jpeg" || img_format_ == "jpg")
      params = {cv::IMWRITE_JPEG_QUALITY, 95}; // Adjust quality as needed
    else if (img_format_ == "png")
      params = {cv::IMWRITE_PNG_COMPRESSION, 3}; // Adjust compression as needed
  
    if (!cv::imencode("." + img_format_, input, buffer, params)) {
      throw std::runtime_error("Failed to encode image to format: " +
                               img_format_);
    }
  
    // Convert the binary buffer to a base64 string
    std::string encodedImage =
        base64::to_base64(std::string(buffer.begin(), buffer.end()));
  
    if (encodedImage.empty()) {
      throw std::runtime_error("Base64 image is empty!");
    }
  
    return encodedImage;
  }

  /**
   * @brief Converts a ROS2 sensor_msgs::msg::Image message to a base64-encoded JPEG string.
   *
   * This function uses cv_bridge to convert a ROS2 image message into an OpenCV image,
   * and then compresses and base64-encodes it using JPEG format.
   *
   * @param ros_image The input ROS2 image message.
   * @return Base64-encoded string of the JPEG-compressed image.
   *
   * @throws std::runtime_error if the image cannot be converted or encoded.
   */
inline std::string convert_msg_to_base64_(const sensor_msgs::msg::Image &ros_image) {
  try {
    // Create a shared_ptr msg from the image
    const auto msg = std::make_shared<sensor_msgs::msg::Image>(ros_image);

    // Convert ROS2 Image message to OpenCV image (in RGB format)
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); // Assuming all incoming ROS msgs have this encoding

    // Convert RGB (ROS standard) to BGR (OpenCV default) to avoid color shift
    cv::Mat bgr_image;
    cv::cvtColor(cv_ptr->image, bgr_image, cv::COLOR_RGB2BGR);

    return convert_mat_to_base64_(bgr_image, img_format_);
  } catch (cv_bridge::Exception &e) {
    throw std::runtime_error(std::string("Error during image conversion: ") + e.what());
  }
}

  /**
   * @brief Utility function to save a ROS image message to disk as a JPEG file.
   *
   * This function can be used for debugging purposes to inspect intermediate image data.
   *
   * @param img ROS sensor_msgs::msg::Image to be saved.
   * @param filename Target path to save the JPEG image (e.g., "/tmp/debug.jpg").
   * @throws std::runtime_error if the image cannot be decoded or saved.
   */
  inline void save_image_to_file(const sensor_msgs::msg::Image &img, const std::string &filename) {
    try {
      const auto msg = std::make_shared<sensor_msgs::msg::Image>(img);
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); //OpenCV expects this encoding

      if (!cv::imwrite(filename, cv_ptr->image)) {
        throw std::runtime_error("Failed to write image to file: " + filename);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Image debug save failed: " + std::string(e.what()));
    }
  }

  /**
   * @brief Utility function to save a base64-encoded image to a binary file for debugging.
   *
   * Decodes the base64 string and writes the resulting binary image content to the file.
   *
   * @param base64_img The base64-encoded image string.
   * @param filename The path to write the decoded image binary to (e.g., "/tmp/image.jpg").
   * @throws std::runtime_error if decoding or writing fails.
   */
  inline void save_image_to_file(const std::string &base64_img, const std::string &filename) {
    try {
      const std::string decoded = base64::from_base64(base64_img);
      std::ofstream out(filename, std::ios::binary);
      if (!out) {
        throw std::runtime_error("Could not open file for writing: " + filename);
      }
      out.write(decoded.data(), decoded.size());
      out.close();
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to save base64 image to file: " + std::string(e.what()));
    }
  }

  /// Default LLM model name (used when no model is specified via input port).
  static constexpr const char *default_llm_model_ = "llava";

  /// Maximum number of tokens allowed in the LLM response.
  const int max_tokens_ = 200;

  /// Image format for encoding
  const std::string img_format_ = "jpeg";

  /// Timeout for preloading model
  const std::chrono::steady_clock::duration preload_timeout_ = std::chrono::minutes(2);

  /// Timeout for preloading model
  const std::chrono::steady_clock::duration inference_timeout_ = std::chrono::minutes(1);

};

} // namespace query_llm_behavior

#endif // QUERY_LLM_BEHAVIOR_HPP
