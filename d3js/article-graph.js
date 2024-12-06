d3.json("hypergraph_data.json").then(function (data) {
    const svg = d3.select("svg");
    const width = +svg.attr("width") || svg.node().getBoundingClientRect().width;
    const height = +svg.attr("height") || svg.node().getBoundingClientRect().height;
    const color = d3.scaleOrdinal(d3.schemeTableau10);

    const centralNode = data.nodes.reduce((maxNode, node) => 
        (node.importance || 0) > (maxNode.importance || 0) ? node : maxNode, data.nodes[0]);

    const MIN_DISTANCE = 100;
    const MAX_DISTANCE = 300;

    data.nodes.forEach((node, index) => {
        if (node === centralNode) {
            node.x = width / 2;
            node.y = height / 2;
            node.fx = width / 2;
            node.fy = height / 2;
        } else {
            const angle = (index / data.nodes.length) * 2 * Math.PI;
            const radius = Math.min(width, height) / 3;
            node.x = width / 2 + radius * Math.cos(angle);
            node.y = height / 2 + radius * Math.sin(angle);
        }
    });

    const simulation = d3.forceSimulation(data.nodes)
        .force("charge", d3.forceCollide().radius(50))
        .force("link", d3.forceLink(data.edges.flatMap(edge => 
            edge.nodes.length === 2 
                ? [{
                    source: data.nodes.find(n => n.id === edge.nodes[0]),
                    target: data.nodes.find(n => n.id === edge.nodes[1])
                }] 
                : edge.nodes.map((nodeId, index) => ({
                    source: data.nodes.find(n => n.id === nodeId),
                    target: data.nodes.find(n => n.id === edge.nodes[(index + 1) % edge.nodes.length])
                }))
        )).distance(200))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("radial", d3.forceRadial(MAX_DISTANCE, width / 2, height / 2))
        .alphaMin(0.1)
        .alphaDecay(0.05);

        function createEdgeTabs() {
            const edgeList = d3.select('#edge-list');
            edgeList.selectAll("*").remove();
        
            const uniqueEdges = Array.from(new Set(data.edges.map(edge => edge.id)));
            
            edgeList.selectAll('button')
                .data(uniqueEdges)
                .enter()
                .append('button')
                .text(edgeId => `Edge ${edgeId}`)
                .attr('class', 'w-full mb-2 px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300')
                .on('click', (event, edgeId) => {
                    // 기존 패널 제거
                    svg.selectAll(".edge-description-panel").remove();
        
                    updateVisualizationForEdge(edgeId);
        
                    // 해당 엣지의 노드들 찾기
                    const edge = data.edges.find(e => e.id === edgeId);
                    const edgeNodes = edge.nodes.map(nodeId => 
                        data.nodes.find(node => node.id === nodeId)
                    );
        
                    // 노드들의 평균 위치 계산
                    const avgX = d3.mean(edgeNodes, node => node.x);
                    const avgY = d3.mean(edgeNodes, node => node.y);
        
                    const descriptionPanel = svg.append("g")
                        .attr("class", "edge-description-panel")
                        .attr("transform", `translate(${avgX + 50}, ${avgY - 100})`);
                    
                    // 흰색 배경의 사각형 추가
                    descriptionPanel.append("rect")
                        .attr("width", 250)
                        .attr("height", 100)
                        .attr("fill", "white")
                        .attr("stroke", color(edgeId))
                        .attr("stroke-width", 2)
                        .attr("rx", 10)
                        .attr("ry", 10);
                    
                    // 엣지 ID 텍스트
                    descriptionPanel.append("text")
                        .attr("x", 10)
                        .attr("y", 30)
                        .attr("font-weight", "bold")
                        .text(`Edge ${edgeId} Description`);
                    
                    // 엣지 설명 텍스트
                    descriptionPanel.append("text")
                        .attr("x", 10)
                        .attr("y", 50)
                        .attr("width", 230)
                        .attr("font-size", "0.8em")
                        .text(edge.description || 'No description available.')
                        .call(wrap, 230);
                });
        
            edgeList.append('button')
                .text('View All')
                .attr('class', 'w-full mb-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600')
                .on('click', () => {
                    svg.selectAll(".edge-description-panel").remove();
                    updateVisualizationForEdge(null);
                });
        }
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
        .attr("stroke", (d, i) => color(d.edge.id))
        .attr("stroke-width", d => (d.edge.importance || 1) * 3)
        .on("click", function(event, d) {
            createEdgeInfoTooltip(event, d);
        });

    let nodeGroup = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(data.nodes)
        .enter().append("g");

    const nodeCircles = nodeGroup.append("circle")
        .attr("r", d => Math.max((d.importance || 1) * 15, 10))
        .attr("fill", "gray")
        .attr("class", "node-circle");

    nodeGroup.append("text")
        .text(d => d.id)
        .attr("fill", "black")
        .attr("dy", -20)
        .style("pointer-events", "none");

    function updateVisualizationForEdge(visibleEdgeId) {
        if (visibleEdgeId === null) {
            nodeGroup.style("opacity", 1);
            edgeLines.style("opacity", 1);
        } else {
            const filteredEdges = data.edges.filter(edge => edge.id === visibleEdgeId);
            const visibleNodeIds = new Set(filteredEdges.flatMap(edge => edge.nodes));

            nodeGroup.style("opacity", d => visibleNodeIds.has(d.id) ? 1 : 0.2);
            edgeLines.style("opacity", edge => 
                filteredEdges.some(filteredEdge => filteredEdge.id === edge.edge.id) ? 1 : 0.2
            );
        }
    }

    createEdgeTabs();

    nodeGroup
        .on("mouseover", function(event, d) {
            const connectedEdges = data.edges.filter(edge => edge.nodes.includes(d.id));

            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", node => {
                    if (node.id === d.id) return Math.max((node.importance || 1) * 25, 20);
                    return Math.max((node.importance || 1) * 15, 10);
                })
                .attr("fill", node => node.id === d.id ? "red" : "gray");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", edge => {
                    if (connectedEdges.includes(edge.edge)) return (edge.edge.importance || 1) * 6;
                    return (edge.edge.importance || 1) * 3;
                })
                .attr("stroke", edge => {
                    if (connectedEdges.includes(edge.edge)) return "red";
                    return color(edge.edge.id);
                });
        })
        .on("mouseout", function() {
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", d => Math.max((d.importance || 1) * 15, 10))
                .attr("fill", "gray");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", d => (d.edge.importance || 1) * 3)
                .attr("stroke", d => color(d.edge.id));
        });

    const drag = d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);

    nodeGroup.call(drag);

    function dragStarted(event, d) {
        if (d === centralNode) return;
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        if (d === centralNode) return;
        const dx = event.x - width / 2;
        const dy = event.y - height / 2;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance >= MIN_DISTANCE && distance <= MAX_DISTANCE) {
            d.fx = Math.max(0.05 * width, Math.min(0.95 * width, event.x));
            d.fy = Math.max(0.05 * height, Math.min(0.95 * height, event.y));
        }
    }

    function dragEnded(event, d) {
        if (d === centralNode) return;
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

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
        
        tooltip.append("h3").text(`Edge ${edgeId} Details`);
        tooltip.append("p").text(description);
        tooltip.append("div")
            .text(`Nodes: ${d.edge.nodes.join(", ")}`)
            .style("font-size", "0.8em")
            .style("color", "gray");
    }

    d3.select("body").on("click", function(event) {
        if (!event.target.closest(".edge-tooltip")) {
            d3.selectAll(".edge-tooltip").remove();
        }
    });

}).catch(function (error) {
    console.error("Error loading data:", error);
});