package slm

// Meta key constants for Message.Meta extension fields.
// These are application-layer metadata that do NOT participate in
// JSON serialization or LLM request construction.

const (
	// MetaImagePaths stores image file paths attached to a user message.
	// Value type: []string
	// Before sending to the LLM, the caller renders these paths into
	// base64 ImagePart (vision model) or inline path text (OCR model).
	MetaImagePaths = "image_paths"

	// MetaWorkingMode stores the working mode tag for a user message.
	// Value type: string (e.g. "karpathy", "openspec", "superpowers")
	// The caller appends mode-specific prompts before sending to the LLM.
	MetaWorkingMode = "working_mode"
)
