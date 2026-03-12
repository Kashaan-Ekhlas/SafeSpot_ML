# SafeSpot Moderation System

SafeSpot is a text moderation pipeline designed for social media platforms.
It combines rule-based filtering, machine learning, and large language models to detect harmful or inappropriate content while still understanding context.

The system uses **multiple layers of moderation**, with each layer handling progressively more complex cases. At each step in the layer, any takedown comes with what policies were violated hence making it explainable at every layer.
| Label         | Description                           |
| ------------- | ------------------------------------- |
| S1_harassment | Abusive or insulting language         |
| S2_hate       | Hate speech targeting identity groups |
| S3_violence   | Threats or promotion of violence      |
| S4_sexual     | Sexual or explicit content            |
| S8_safe       | Non-toxic or acceptable content       |




# Moderation Pipeline

## Layer 1 — Basic Word Filter

The first layer is a simple rule-based filter.

It removes obvious slurs, banned phrases, and clearly harmful words using a predefined list. This layer is extremely fast O(k) where k = number of words in a post, serves as a first line of defence to avoid usage of heavier, expensive models.


## Layer 2 — DeBERTa Classification

The second layer is a machine learning classifier built using **DeBERTa-v3-base**, fine-tuned for multi-label content moderation.

This model analyzes the text and predicts whether it violates any moderation policies. Because it uses a transformer architecture, it can understand tone and sentence structure better than simple keyword filters.

The classifier predicts probabilities for several moderation categories, such as harassment, hate speech, violence, and sexual content.

If the model is confident in its decision, the system can automatically approve or flag the content.



## Layer 3 — Context Review with Llama

If the DeBERTa model is uncertain, the message moves to the third layer.

This layer uses **Llama 8B** to evaluate the content with deeper contextual reasoning. The model can consider surrounding or parent context while deciding a label.

This helps handle cases where the same wording could be harmless in one context but inappropriate in another.

Example:

Post about operating systems
Comment:
“Tell me about the kill process command.”

→ Technical discussion, safe.

Post about violence
Comment:
“Tell me about the kill process.”

→ Potentially inappropriate.

Llama also identifies which moderation category the message belongs to if it violates a policy.


## Final Layer — Human Moderation

No automated system is perfect. The final safeguard is human moderation. Human touch is never replicable!

Messages that remain ambiguous or sensitive can be forwarded to moderators for manual review. This ensures that complex or nuanced situations are handled responsibly.



# Goal of the System

The goal of this layered approach is to balance **speed, accuracy, explainability, and context awareness**.

* The word filter handles obvious violations instantly.
* DeBERTa provides fast large-scale classification.
* Llama handles complex contextual cases.
* Human moderators make final decisions when necessary.

Together, these layers form a scalable moderation system suitable for modern social platforms.
