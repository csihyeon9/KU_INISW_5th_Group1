// hypergraph_visualization.js
// Hypergraph 데이터 로드
function visualizeHypergraph (data) {
    const svg = d3.select("svg");
    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    svg.selectAll("*").remove();

    const color = d3.scaleOrdinal(d3.schemeTableau10);

    // 가장 중요도가 큰 노드 찾기
    const centralNode = data.nodes.reduce((maxNode, node) => (node.importance || 0) > (maxNode.importance || 0) ? node : maxNode, data.nodes[0]);

    // 중심 노드로부터의 최소/최대 거리 제한
    const MIN_DISTANCE = 100;  // 중심으로부터 최소 거리
    const MAX_DISTANCE = 300;  // 중심으로부터 최대 거리

    // Force Simulation 설정
    const simulation = d3.forceSimulation(data.nodes)
        .force("charge", d3.forceCollide().radius(50))
        .force("link", d3.forceLink(data.edges.flatMap(edge => 
            edge.nodes.length === 2 ? [{
                source: data.nodes.find(n => n.id === edge.nodes[0]),
                target: data.nodes.find(n => n.id === edge.nodes[1])
            }] : edge.nodes.map((nodeId, index) => ({
                source: data.nodes.find(n => n.id === nodeId),
                target: data.nodes.find(n => n.id === edge.nodes[(index + 1) % edge.nodes.length])
            }))
        )).distance(10))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("radial", d3.forceRadial(MAX_DISTANCE, width / 2, height / 2))
        .force("boundary", () => {
            data.nodes.forEach(node => {
                if (node !== centralNode) {
                    node.x = Math.max(50, Math.min(width - 50, node.x));
                    node.y = Math.max(50, Math.min(height - 50, node.y));
                }
            });
        });

    // 중앙 노드 위치 고정
    centralNode.fx = width / 2;
    centralNode.fy = height / 2;

    // 하이퍼엣지 연결선 (하이퍼엣지로 이루어진 노드들의 중앙으로 연결)
    let edgeLines = svg.append("g")
    .attr("class", "edges")
    .selectAll("line")
    .data(data.edges.flatMap(d => {
        if (d.nodes.length === 2) {
            return [{ edge: d, nodeId1: d.nodes[0], nodeId2: d.nodes[1] }];
        } else {
            return d.nodes.map(nodeId => ({ edge: d, nodeId: nodeId }));
        }
    }))
    .enter()
    .append("line")
    .attr("stroke", "gray")          // 모든 엣지 회색으로 통일
    .attr("stroke-width", 1)         // 모든 엣지 너비 1로 통일
    .on("click", function(event, d) {
        const edgeId = d.edge.id;
        const url = `edge_info.html?edgeId=${edgeId}`;
        window.open(url, '_blank');
        createEdgeInfoTooltip(event, d);
    });

    // 노드 그룹 컨테이너 생성
    let nodeGroup = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(data.nodes)
        .enter().append("g");

    // 하이퍼엣지 경계박스 생성 함수
    function createHyperedgeBoundingBox(edge) {
        // 해당 엣지의 노드들 찾기
        const nodeCircles = nodeGroup.append("circle")
            .attr("r", d => {
                // 중앙 노드인 경우 더 큰 크기 적용
                if (d === centralNode) {
                    return Math.max((d.importance || 1) * 30, 20);  // 중앙 노드는 30배, 최소 20
                }
                // 일반 노드
                return Math.max((d.importance || 1) * 15, 10);  // 일반 노드는 15배, 최소 10
            })
            .attr("fill", "#32CD32")
            .attr("class", "node-circle");

        const edgeNodes = edge.nodes.map(nodeId => 
            data.nodes.find(node => node.id === nodeId)
        );

        // 노드들의 좌표 계산
        const xCoords = edgeNodes.map(node => node.x);
        const yCoords = edgeNodes.map(node => node.y);

        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        // 여유 공간 추가
        const padding = 30;

        // 경계 박스 생성
        const boundingBox = svg.append("rect")
            .attr("class", "hyperedge-bounding-box")
            .attr("x", minX - padding)
            .attr("y", minY - padding)
            .attr("width", maxX - minX + 2 * padding)
            .attr("height", maxY - minY + 2 * padding)
            .attr("fill", "none")
            .attr("stroke", color(edge.id))
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "10,5");

        return boundingBox;
    }

    // 노드 (원)
    const nodeCircles = nodeGroup.append("circle")
        .attr("r", d => d === centralNode 
            ? Math.max((d.importance || 1) * 30, 25)  // 중앙 노드는 항상 큼
            : Math.max((d.importance || 1) * 15, 10)  // 일반 노드는 더 작음
        )
        .attr("fill", "#32CD32")
        .attr("class", "node-circle");

    // 노드 레이블 (제목) - 텍스트 크기를 노드 크기에 비례하게 설정
    const nodeLabels = nodeGroup.append("text")
        .text(d => d.id)
        .attr("fill", "black")
        .attr("dy", d => d === centralNode ? 5 : -20) 
        .attr("text-anchor", "middle")
        .style("pointer-events", "none")
        .style("font-size", d => d === centralNode ? "14px" : "12px")
        // .attr("fill", d => d === centralNode ? "#FFFFFF" : "#000000")
        .attr("font-weight", d => d === centralNode ? "bold" : "normal")
        .style("font-family", "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif");

    function updateVisualizationForEdge(clickedEdgeId) {
        // 이미 선택된 엣지를 다시 클릭한 경우 선택 해제
        if (selectedEdgeId === clickedEdgeId) {
            selectedEdgeId = null;
            // 모든 노드와 엣지를 원래 상태로 복원
            nodeGroup.style("opacity", 1);
            edgeLines.style("opacity", 1);
            // 경계 박스 제거
            svg.selectAll(".hyperedge-bounding-box").remove();
            // 설명 패널 제거
            svg.selectAll(".edge-description-panel").remove();
            return;
        }

        // 새로운 엣지 선택
        selectedEdgeId = clickedEdgeId;
        
        // 특정 엣지와 연결된 노드만 표시
        const filteredEdges = data.edges.filter(edge => edge.id === selectedEdgeId);
        const visibleNodeIds = new Set(filteredEdges.flatMap(edge => edge.nodes));

        // 노드 필터링
        nodeGroup.style("opacity", d => visibleNodeIds.has(d.id) ? 1 : 0.2);

        // 엣지 필터링
        edgeLines.style("opacity", edge => 
            filteredEdges.some(filteredEdge => filteredEdge.id === edge.edge.id) ? 1 : 0.2
        );
    }

    // 범례 생성
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 150}, 20)`);

    const uniqueEdges = Array.from(new Set(data.edges.map(edge => edge.id)));
    
    // 범례 아이템 생성
    uniqueEdges.forEach((edgeId, index) => {
        const edge = data.edges.find(e => e.id === edgeId);
        
        const legendItem = legend.append("g")
            .attr("transform", `translate(0, ${index * 25})`)
            .attr("class", "legend-item")
            .style("cursor", "pointer")
            .on("click", () => {
                // 기존 경계 박스와 설명 패널 제거
                svg.selectAll(".hyperedge-bounding-box").remove();
                svg.selectAll(".edge-description-panel").remove();

                // 엣지 필터링 업데이트
                updateVisualizationForEdge(edgeId);

                // 선택된 상태이고 3개 이상의 노드를 가진 하이퍼엣지인 경우 경계 박스 생성
                if (selectedEdgeId === edgeId && edge.nodes.length >= 3) {
                    createHyperedgeBoundingBox(edge);
                }

                // 선택된 상태인 경우에만 설명 패널 생성
                if (selectedEdgeId === edgeId) {
                    const descriptionPanel = svg.append("g")
                        .attr("class", "edge-description-panel")
                        .attr("transform", `translate(${width - 300}, ${height - 150})`);
                    
                    descriptionPanel.append("rect")
                        .attr("width", 250)
                        .attr("height", 50)
                        .attr("fill", "white")
                        .attr("stroke", color(edgeId));
                    
                    descriptionPanel.append("text")
                        .attr("x", 10)
                        .attr("y", 20)
                        .attr("font-weight", "bold")
                        .text(`${edgeId} Description`);
                    
                    descriptionPanel.append("text")
                        .attr("x", 10)
                        .attr("y", 40)
                        .attr("width", 230)
                        .attr("font-size", "0.8em")
                        .text(edge.description || 'No description available.');
                }
            });

        legendItem.append("circle")
            .attr("r", 8)
            .attr("fill", color(edgeId))
            .attr("cx", 10)
            .attr("cy", 10);

        legendItem.append("text")
            .attr("x", 40)
            .attr("y", 15)
            .text(`${edge.category}`);
    });

    // 전체 그래프 보기 클릭 원 추가
    const resetLegendItem = legend.append("g")
        .attr("transform", `translate(0, ${uniqueEdges.length * 25})`)
        .attr("class", "legend-item")
        .style("cursor", "pointer")
        .on("click", () => {
            // 경계 박스 제거
            svg.selectAll(".hyperedge-bounding-box").remove();
        });

    // resetLegendItem.append("circle")
    //     .attr("r", 8)
    //     .attr("fill", "grey")
    //     .attr("cx", 10)
    //     .attr("cy", 10);

    // resetLegendItem.append("text")
    //     .attr("x", 40)
    //     .attr("y", 15)
    //     .text("View All");

    // 마우스 오버/아웃 이벤트
    nodeGroup
        .on("mouseover", function(event, d) {
            const connectedEdges = data.edges.filter(edge => edge.nodes.includes(d.id));

            // 노드 크기 변경
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", node => {
                    if (node === centralNode) {
                        // 중앙 노드는 hover된 경우에만 더 커짐
                        return node === d ? Math.max((node.importance || 1) * 35, 30) : Math.max((node.importance || 1) * 30, 25);
                    }
                    // 호버된 일반 노드
                    if (node.id === d.id) {
                        return Math.max((node.importance || 1) * 20, 15);
                    }
                    // 나머지 일반 노드
                    return Math.max((node.importance || 1) * 15, 10);
                });

            // 텍스트 크기 변경
            nodeGroup.selectAll("text")
                .transition()
                .duration(200)
                .style("font-size", node => {
                    if (node === centralNode) {
                        // 중앙 노드 텍스트는 hover된 경우에만 더 커짐
                        return node === d ? "18px" : "16px";
                    }
                    // 호버된 일반 노드
                    if (node.id === d.id) {
                        return "14px";
                    }
                    // 나머지 일반 노드
                    return "12px";
                });

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", edge => {
                    if (connectedEdges.includes(edge.edge)) return 2;  // 연결된 엣지만 살짝 굵게
                    return 1;
                })
                .attr("stroke", "gray");  // 색상은 항상 회색 유지
        })
        .on("mouseout", function() {
            // 노드 크기 원복
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", node => node === centralNode 
                    ? Math.max((node.importance || 1) * 30, 25)  // 중앙 노드 기본 크기
                    : Math.max((node.importance || 1) * 15, 10)  // 일반 노드 기본 크기
                );

            // 텍스트 크기 원복
            nodeGroup.selectAll("text")
                .transition()
                .duration(200)
                .style("font-size", node => node === centralNode ? "16px" : "12px");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", 1)
                .attr("stroke", "gray");
        })

        .on("mouseout", function() {
            // 노드 크기 원복
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", d => d === centralNode 
                    ? Math.max((d.importance || 1) * 30, 25)  // 중앙 노드 크기 유지
                    : Math.max((d.importance || 1) * 15, 10)  // 일반 노드 원래 크기로
                );

            // 텍스트 크기 원복
            nodeGroup.selectAll("text")
                .transition()
                .duration(200)
                .style("font-size", d => d === centralNode ? "16px" : "12px");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", 1)
                .attr("stroke", d => 'gray');
        });

    // 드래그 이벤트 설정
    const drag = d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);

    nodeGroup.call(drag);

    function dragStarted(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 시뮬레이션 일시 중지
        if (!event.active) simulation.alphaTarget(0.3).restart();
        
        // 노드 위치 고정
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 중심 노드로부터의 거리 제한
        const dx = event.x - width / 2;
        const dy = event.y - height / 2;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // 최소 및 최대 거리 제한
        if (distance >= MIN_DISTANCE && distance <= MAX_DISTANCE) {
            // 화면 경계 내로 제한
            d.fx = Math.max(0.05 * width, Math.min(0.95 * width, event.x));
            d.fy = Math.max(0.05 * height, Math.min(0.95 * height, event.y));
        }
    }

    function dragEnded(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 시뮬레이션 재개
        if (!event.active) simulation.alphaTarget(0);

        // 드래그 종료 시 위치 고정 해제 (자유롭게 움직일 수 있도록)
        d.fx = null;
        d.fy = null;
    }

    // 위치 업데이트 및 시뮬레이션 이벤트
    simulation.on("tick", () => {
        nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);

        edgeLines
            .attr("x1", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node1 = data.nodes.find(n => n.id === d.nodeId1);
                    return node1.x;
                } else {
                    const edgeNodes = d.edge.nodes.map(nodeId => data.nodes.find(n => n.id === nodeId));
                    return d3.mean(edgeNodes, node => node.x);
                }
            })
            .attr("y1", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node1 = data.nodes.find(n => n.id === d.nodeId1);
                    return node1.y;
                } else {
                    const edgeNodes = d.edge.nodes.map(nodeId => data.nodes.find(n => n.id === nodeId));
                    return d3.mean(edgeNodes, node => node.y);
                }
            })
            .attr("x2", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node2 = data.nodes.find(n => n.id === d.nodeId2);
                    return node2.x;
                } else {
                    const node = data.nodes.find(n => n.id === d.nodeId);
                    return node.x;
                }
            })
            .attr("y2", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node2 = data.nodes.find(n => n.id === d.nodeId2);
                    return node2.y;
                } else {
                    const node = data.nodes.find(n => n.id === d.nodeId);
                    return node.y;
                }
            });
    });

    // Edge descriptions object
    const edgeDescriptions = {
        '1': 'This edge represents a key relationship between connected nodes, highlighting their interconnectedness.',
        '2': 'A complex interconnection showcasing the multifaceted nature of the nodes involved.',
        '3': 'An important structural connection that bridges multiple nodes in the hypergraph.'
    };

    // Modify the existing code to include edge descriptions
    function createEdgeInfoTooltip(event, d) {
        const edgeId = d.edge.id;
        const description = d.edge.description || 'No additional description available.';
        
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "edge-tooltip")
            .style("position", "absolute")
            .style("background", "white")
            .style("border", "1px solid black")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px");
        
        tooltip.append("h3")
            .text(`Edge ${edgeId} Details`);
        
        tooltip.append("p")
            .text(description);
        
        tooltip.append("div")
            .text(`Nodes: ${d.edge.nodes.join(", ")}`)
            .style("font-size", "0.8em")
            .style("color", "gray");
    }

    // Optional: Add a way to close the tooltip
    d3.select("body").on("click", function(event) {
        if (!event.target.closest(".edge-tooltip")) {
            d3.selectAll(".edge-tooltip").remove();
        }
    });

}