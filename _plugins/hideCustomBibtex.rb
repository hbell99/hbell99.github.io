 module Jekyll
  module HideCustomBibtex
    def hideCustomBibtex(input)
	  keywords = @context.registers[:site].config['filtered_bibtex_keywords']

	  keywords.each do |keyword|
		# Only strip lines where `keyword` is the field name (before `=`) — matching
		# it anywhere in the line also caught field *values* that happen to contain
		# the same substring, e.g. a `url` field whose value contains "openreview".
		input = input.gsub(/^\s*#{keyword}\s*=.*$\n/, '')
	  end

      return input
    end
  end
end

Liquid::Template.register_filter(Jekyll::HideCustomBibtex)
