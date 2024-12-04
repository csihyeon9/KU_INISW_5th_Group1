// sidepanel.js
let currentResizeObserver = null;

document.addEventListener('DOMContentLoaded', () => {
    const contentDiv = document.getElementById('content');
    const loadingDiv = document.getElementById('loading');
    let initialMessage = '텍스트를 드래그하면 나타나는 아이콘을 클릭해주세요.';

    resetButton.addEventListener('click', () => {
        chrome.storage.local.remove(['lastAnalysis', 'graphData'], function() {
            contentDiv.innerHTML = initialMessage;
        });
    });

    chrome.storage.local.get(['lastAnalysis', 'graphData'], function(result) {
        if (result.lastAnalysis && result.graphData) {
            contentDiv.innerHTML = result.lastAnalysis;
            
            const graphContainer = document.getElementById('graph-container');
            if (graphContainer) {
                if (currentResizeObserver) {
                    currentResizeObserver.disconnect();
                }
                
                currentResizeObserver = new ResizeObserver(entries => {
                    for (let entry of entries) {
                        if (entry.target.id === 'graph-container') {
                            visualizeHypergraph(result.graphData);
                        }
                    }
                });
                currentResizeObserver.observe(graphContainer);
    
                requestAnimationFrame(() => {
                    visualizeHypergraph(result.graphData);
                });
            }
        } else {
            contentDiv.innerHTML = initialMessage;
        }
    });

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "analyzeText") {
            sendResponse({ received: true });
            handleAnalysis(request.text);
            return false;
        }
    });

    async function handleAnalysis(text) {
        try {
            contentDiv.classList.add('loading-active');
            loadingDiv.style.display = 'block';
            contentDiv.innerHTML = ''; // 로딩 중에는 내용을 비움
            // const response = await fetch('https://finwise.p-e.kr:8000/service', {
            const response = await fetch('http://127.0.0.1:8000/service', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received data:', data);

            if (currentResizeObserver) {
                currentResizeObserver.disconnect();
            }

            currentResizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target.id === 'graph-container') {
                        visualizeHypergraph(data.hypergraph_data);
                    }
                }
            });
            
            const formatDate = (dateStr) => {
                const cleanedDateStr = dateStr
                    .replace('T', ' ')
                    .split('.')[0];
                return cleanedDateStr;
            };
            
            const analysisHTML = `
                <h3>기사 분석 그래프</h3>
                <div id="graph-container">
                    <svg></svg>
                </div>
                <div class="recommendation-section">
                    <h3>관련 기사에서 사용된 키워드도 함께 확인해 보세요.</h3>
                    ${data.recommendations.map(rec => `
                        <div class="recommendation-item">
                            <h4>${rec.keyword} <span class="similarity">(유사도: ${(rec.similarity * 100).toFixed(2)}%)</span></h4>
                            ${rec.articles.length > 0 ? `
                                <ul class="article-list">
                                    ${rec.articles.map(article => `
                                        <li class="article-item">
                                            <strong>${article.title}</strong><br>
                                            <small>URL: ${article.url}</small><br>
                                            <small>날짜: ${article.date}</small><br>
                                            <small>키워드: ${article.keywords.join(', ')}</small>
                                        </li>
                                    `).join('')}
                                </ul>
                            ` : '<p>관련 기사가 없습니다.</p>'}
                        </div>
                    `).join('')}
                </div>
                <small>응답 시간: ${formatDate(data["response_time"])}</small>
            `;
            
            contentDiv.innerHTML = analysisHTML;

            const graphContainer = document.getElementById('graph-container');
            currentResizeObserver.observe(graphContainer);

            requestAnimationFrame(() => {
                visualizeHypergraph(data.hypergraph_data);
            });

            chrome.storage.local.set({
                lastAnalysis: analysisHTML,
                graphData: data.hypergraph_data
            });

        } catch (error) {
            const errorHTML = `<div class="error">분석 중 오류가 발생했습니다: ${error.message}</div>`;
            contentDiv.innerHTML = errorHTML;
            chrome.storage.local.set({
                lastAnalysis: errorHTML
            });
            console.error('Analysis error:', error);
        } finally {
            loadingDiv.style.display = 'none';
            contentDiv.classList.remove('loading-active');
        }
    }
});