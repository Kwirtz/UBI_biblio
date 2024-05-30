install.packages("stargazer")
library(stargazer)


read.csv("C:\\Users\\Eva Jacob\\Documents\\Thèse\\Papier 3\\Data\\table1.csv")

citation <- read.csv("C:\\Users\\Eva Jacob\\Documents\\Thèse\\Papier 3\\Data\\table1.csv")


#Table authors and references
citation_modified <- citation[, c(1, 6, 12, 16)]

num_columns <- ncol(citation_modified)

if (num_columns >= 4) {
  col_4 <- citation_modified[, 4]
  remaining_columns <- citation_modified[, -4]
  citation_modified <- cbind(remaining_columns[, 1, drop=FALSE], col_4, remaining_columns[, 2:(num_columns - 1)])
  colnames(citation_modified)[2] <- colnames(citation_modified)[4]
  print(colnames(citation_modified))
  
  colnames(citation_modified)[1:4] <- c("Community", "Disciplines", "Most Cited References", "Most Cited Authors")
  stargazer(citation_modified, summary = FALSE, rownames = FALSE)
} else {
  stop("Le dataframe 'citation_modified' doit avoir au moins 4 colonnes.")
}

stargazer_table_authors_references <- stargazer(citation_modified, summary = FALSE, rownames = FALSE, type = 'text',
                             out = "C:/Users/Eva Jacob/Documents/Github/UBI_biblio/Results/Table/Table_authors_references.txt")




#Table keywords: gram
citation_terms <- citation[, c(1, 13)]
group_size <- 10
total_rows <- nrow(citation_terms)
num_groups <- ceiling(total_rows / group_size)
combine_rows <- function(df) {
  apply(df, 2, function(column) paste(column, collapse = ","))
}
combined_groups <- list()

for (i in 1:num_groups) {
  start_index <- (i - 1) * group_size + 1
  end_index <- min(i * group_size, total_rows)
  
  group_df <- citation_terms[start_index:end_index, ]

  combined_groups[[i]] <- combine_rows(group_df)
}

citation_gram <- do.call(rbind, combined_groups)

colnames(citation_gram)[1:2] <- c("Community", "10 most distinctive words")

stargazer(citation_gram, summary = FALSE, rownames = FALSE)



#Table keywords: bigram
citation_terms <- citation[, c(1, 14)]
group_size <- 10
total_rows <- nrow(citation_terms)
num_groups <- ceiling(total_rows / group_size)
combine_rows <- function(df) {
  apply(df, 2, function(column) paste(column, collapse = ","))
}
combined_groups <- list()

for (i in 1:num_groups) {
  start_index <- (i - 1) * group_size + 1
  end_index <- min(i * group_size, total_rows)
  
  group_df <- citation_terms[start_index:end_index, ]
  
  combined_groups[[i]] <- combine_rows(group_df)
}

citation_bigram <- do.call(rbind, combined_groups)

colnames(citation_bigram)[1:2] <- c("Community", "10 most distinctive groups of two words")

stargazer(citation_bigram, summary = FALSE, rownames = FALSE)



#Table keywords: trigram
citation_terms <- citation[, c(1, 15)]
group_size <- 10
total_rows <- nrow(citation_terms)
num_groups <- ceiling(total_rows / group_size)
combine_rows <- function(df) {
  apply(df, 2, function(column) paste(column, collapse = ","))
}
combined_groups <- list()

for (i in 1:num_groups) {
  start_index <- (i - 1) * group_size + 1
  end_index <- min(i * group_size, total_rows)
  
  group_df <- citation_terms[start_index:end_index, ]
  
  combined_groups[[i]] <- combine_rows(group_df)
}

citation_trigram <- do.call(rbind, combined_groups)

colnames(citation_trigram)[1:2] <- c("Community", "10 most distinctive groups of three words")

stargazer(citation_trigram, summary = FALSE, rownames = FALSE)

