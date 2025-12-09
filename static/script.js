document.getElementById("startLive").onclick = () => {
    const liveImg = document.getElementById("liveFeed");
    liveImg.src = "/live";
    liveImg.style.display = "block";
};
