document.addEventListener("mouseup", (event) => {
    const selection = window.getSelection().toString();

    if (selection) {
        showIcon(event, selection);
    }
});

function showIcon(event, draggedText) {
    let icon = document.createElement("div");
    icon.innerHTML = "🔍"; // 아이콘 표시
    icon.style.position = "absolute";
    icon.style.left = `${event.pageX}px`;
    icon.style.top = `${event.pageY}px`;
    icon.style.cursor = "pointer";
    document.body.appendChild(icon);

    // 아이콘 클릭 시 메시지 전송
    icon.addEventListener("click", () => {
        chrome.runtime.sendMessage({ action: "openPanel", text: draggedText });
    });

    // 3초 후 아이콘 숨기기
    setTimeout(() => {
        document.body.removeChild(icon);
    }, 3000);
}
