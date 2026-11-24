# slugify

`slugify("Hello, World!")` returns `"hello-world"`.

Rules: lowercase; each run of non-alphanumeric ASCII characters becomes one hyphen; no
leading or trailing hyphen; `""` when nothing is left.
