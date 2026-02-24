function copyBibtex() {
  var bibtexText = document.getElementById("bibtexContent").innerText;
  navigator.clipboard.writeText(bibtexText).then(function () {
    var btn = document.getElementById("copyButton");
    btn.innerText = "Copied!";
    setTimeout(function () {
      btn.innerText = "Copy";
    }, 2000);
  });
}
