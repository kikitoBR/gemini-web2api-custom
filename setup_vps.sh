#!/bin/bash

# Configurações de cores para o terminal
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # Sem cor

echo -e "${YELLOW}====================================================${NC}"
echo -e "${GREEN}    Instalador Automático - Gemini Web2API (VPS)     ${NC}"
echo -e "${YELLOW}====================================================${NC}"

# 1. Verificar se está rodando como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Erro: Por favor, execute este script como root ou usando sudo.${NC}"
  exit 1
fi

# 2. Atualizar repositórios do sistema
echo -e "\n${YELLOW}[1/6] Atualizando pacotes do sistema...${NC}"
apt update && apt upgrade -y

# 3. Instalar dependências básicas (curl, git)
echo -e "\n${YELLOW}[2/6] Instalando dependências básicas (git, curl, jq)...${NC}"
apt install -y git curl jq

# 4. Instalar Docker e Docker Compose se não estiverem presentes
if ! command -v docker &> /dev/null; then
    echo -e "\n${YELLOW}[3/6] Docker não encontrado. Instalando Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    echo -e "\n${GREEN}[3/6] Docker já está instalado!${NC}"
fi

# 5. Clonar ou atualizar o repositório
INSTALL_DIR="/opt/gemini-web2api"
echo -e "\n${YELLOW}[4/6] Configurando repositório em ${INSTALL_DIR}...${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Diretório já existe. Atualizando código...${NC}"
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/Sophomoresty/gemini-web2api.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 6. Configurar o arquivo config.json
echo -e "\n${YELLOW}[5/6] Configurando chaves de segurança (API Keys)...${NC}"
if [ ! -f "config.json" ]; then
    # Gerar uma chave de API segura por padrão se o usuário não quiser definir uma
    DEFAULT_KEY="sk-gemini-$(openssl rand -hex 12)"
    
    echo -e "Digite a chave de API que você deseja usar (Pressione Enter para usar a chave gerada automaticamente: ${GREEN}${DEFAULT_KEY}${NC}):"
    read -r USER_KEY
    
    if [ -z "$USER_KEY" ]; then
        FINAL_KEY="$DEFAULT_KEY"
    else
        FINAL_KEY="$USER_KEY"
    fi

    # Criar config.json a partir do exemplo
    cp config.example.json config.json
    
    # Atualizar o arquivo config.json com a chave de API fornecida usando jq
    jq --arg key "$FINAL_KEY" '.api_keys = [$key]' config.example.json > config.json
    
    echo -e "${GREEN}Configuração salva com sucesso com a chave de API: ${FINAL_KEY}${NC}"
else
    echo -e "${GREEN}Arquivo config.json já existe. Mantendo configurações atuais.${NC}"
    FINAL_KEY=$(jq -r '.api_keys[0]' config.json)
fi

# 7. Iniciar os serviços com Docker Compose
echo -e "\n${YELLOW}[6/6] Iniciando o Gemini Web2API no Docker...${NC}"
docker compose -f docker-compose.local.yml down &>/dev/null
docker compose -f docker-compose.local.yml up -d --build

# 8. Validar instalação
echo -e "\n${YELLOW}Aguardando o serviço inicializar...${NC}"
sleep 3

if docker compose -f docker-compose.local.yml ps | grep -q "Up"; then
    VPS_IP=$(curl -s https://api.ipify.org)
    echo -e "\n${GREEN}====================================================${NC}"
    echo -e "${GREEN}🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!${NC}"
    echo -e "${GREEN}O Gemini Web2API está online, rodando 24/7 e iniciará junto com o sistema!${NC}"
    echo -e "${YELLOW}====================================================${NC}"
    echo -e "🔗 URL Base da API: ${CYAN}http://${VPS_IP}:8081/v1${NC}"
    echo -e "🔑 Sua Chave de API: ${GREEN}${FINAL_KEY}${NC}"
    echo -e "\n${YELLOW}Para testar se está funcionando, você pode rodar:${NC}"
    echo -e "curl http://localhost:8081/v1/chat/completions -H \"Content-Type: application/json\" -H \"Authorization: Bearer ${FINAL_KEY}\" -d '{\"model\":\"gemini-3.5-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Olá\"}]}'"
    echo -e "${YELLOW}====================================================${NC}"
else
    echo -e "\n${RED}Erro: O contêiner Docker não pôde ser iniciado. Verifique os logs usando:${NC}"
    echo -e "docker compose -f docker-compose.local.yml logs"
fi
