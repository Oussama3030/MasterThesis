export OPTIONS="-b --configuration json://Config.json"

o2-analysis-em-my-track-select ${OPTIONS} | o2-analysis-timestamp ${OPTIONS} | o2-analysis-trackselection ${OPTIONS} | o2-analysis-track-propagation ${OPTIONS}
