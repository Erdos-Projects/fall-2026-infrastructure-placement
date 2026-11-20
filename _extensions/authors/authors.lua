-- {{< authors >}}: the repository link and author cards for the sidebar,
-- built from the `team` list in authors.yml.

local str = pandoc.utils.stringify

-- One icon link, skipped when the author has no URL for it
local function link(url, icon, label)
  if not url then return "" end
  return string.format(
    '<a href="%s" target="_blank"><i class="bi bi-%s"></i> %s</a>',
    str(url), icon, label)
end

return {
  ["authors"] = function(args, kwargs, meta)
    local html = {
      '<div class="author-profile">',
      '<div class="project-repo"><a href="' .. str(meta["repo-url"]) .. '" target="_blank">',
      '<i class="bi bi-github"></i> Access Project Repository</a></div>',
      '<hr>',
      '<h3 class="authors-heading">Authors</h3>',
    }
    for _, a in ipairs(meta.team or {}) do
      table.insert(html, '<div class="author-card">')
      if a.image then
        table.insert(html, string.format('<img src="%s" alt="%s" class="author-image">',
          str(a.image), str(a.alt or a.name)))
      end
      table.insert(html, '<h4>' .. str(a.name) .. '</h4>')
      table.insert(html, '<div class="author-links">'
        .. link(a.github, "github", "GitHub")
        .. link(a.linkedin, "linkedin", "LinkedIn")
        .. link(a.website, "globe", "Website")
        .. link(a.scholar, "journal-text", "Research")
        .. '</div></div>')
    end
    table.insert(html, '</div>')
    return pandoc.RawBlock("html", table.concat(html, "\n"))
  end
}
