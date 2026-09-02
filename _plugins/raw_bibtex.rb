# Exposes the verbatim, as-typed text of each _bibliography/papers.bib entry
# as site.data['raw_bibtex'][citation_key], so the "Copy Bib" button can copy
# exactly what's in the source file instead of jekyll-scholar/bibtex-ruby's
# re-serialized form (which normalizes author name order to "Last, First"
# and drops fields like `url` entirely).
module Jekyll
  class RawBibtexGenerator < Generator
    priority :high

    def generate(site)
      path = File.join(site.source, "_bibliography", "papers.bib")
      return unless File.exist?(path)

      content = File.read(path)
      content = content.sub(/\A---\s*\n---\s*\n/, "")

      entries = {}
      current_key = nil
      buffer = []

      content.each_line do |line|
        if line =~ /\A@[a-zA-Z]+\{\s*([^,\s]+)\s*,/
          entries[current_key] = buffer.join.strip if current_key
          current_key = Regexp.last_match(1)
          buffer = [line]
        elsif current_key
          buffer << line
        end
      end
      entries[current_key] = buffer.join.strip if current_key

      site.data["raw_bibtex"] = entries
    end
  end
end
