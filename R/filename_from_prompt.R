#` filename_from_prompt` function
#' @title Generate a filename from a prompt
#' @description This function generates a filename from a prompt by removing all non-alphanumeric characters and replacing them with underscores. The filename is limited to 50 characters. If `datetime` is set to TRUE, the current date and time are prepended to the filename.
#``
#' @param prompt A character string representing the prompt.
#' @param datetime Logical indicating whether to prepend the current date and time to the filename. Default is TRUE.
#' @return A character string representing the generated filename.
#'
#' @examples
#' filename_from_prompt("A beautiful sunset over the mountains")
#' filename_from_prompt("A beautiful sunset over the mountains", datetime = FALSE)
#' @export
filename_from_prompt <- function(
  prompt,
  datetime = TRUE
) {
  # Remove all non-alphanumeric characters from the prompt
  prompt_strs <- gsub("[^a-zA-Z0-9]", "_", prompt)
  # limit prompt length to 50 characters
  prompt_strs <- substr(prompt_strs, 1, 50)
  if (datetime == FALSE) {
    # Create a filename using the modified prompt
    file_name <- paste0("prompt_", prompt_strs, ".png")
  } else {
    # date time stamp
    datetime <- format(Sys.time(), "%Y%m%d_%H%M%S")
    file_name <- paste0(datetime, "_", prompt_strs, ".png")
  }
  return(file_name)
}

