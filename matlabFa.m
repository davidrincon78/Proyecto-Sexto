

% =========================================================================
% SCRIPT DE MATLAB: Caracterizacion de Sena (Media + Desviacion + Fases)
%
% Flujo:
%   1. Seleccionas la carpeta de UNA sena (ej. resultados/ACCIDENTE) o la
%      carpeta 'resultados' completa (todas las senas). El script recorre las
%      subcarpetas (Todos_rep1, Todos_rep2, ...) y de cada una toma solo
%      'angulos.xlsx' y 'fases.xlsx'. Cada sena se caracteriza por separado y
%      sus figuras / Excel se guardan en la carpeta de esa sena. Con varias
%      senas se escribe ademas 'resumen_todas_las_senas.xlsx'.
%   2. De cada fases.xlsx lee: limites de cada fase (hoja 'Fases'),
%      nombre de la sena y brazo dominante (hoja 'Info').
%   3. Recorta cada repeticion al tramo activo del gesto (inicio de F1 a
%      fin de la ultima fase) y lo normaliza a 0-100 %.
%      - modo 'fases' : normalizacion POR TRAMOS. Cada fase de cada
%                       repeticion se estira a su % promedio, de modo que
%                       los limites de fase coinciden en todas.
%      - modo 'global': normalizacion lineal unica (como la version anterior).
%   4. Calcula punto a punto media y desviacion estandar de codo, hombro y
%      flexion de dedos del brazo dominante y las grafica con las franjas
%      de fase (sostenimiento / movimiento).
%   5. Guarda en la misma carpeta la figura (PNG) y un Excel de resumen.
%   Si las repeticiones tienen distinta secuencia de fases (variantes de
%   ejecucion), se genera una figura y un Excel por variante
%   (caracterizacion_<SENA>_<K>fases.png, resumen_<SENA>_<K>fases.xlsx).
%
% Requiere MATLAB R2019b o superior.
% =========================================================================
clear; clc; close all;

%% 0. CONFIGURACION -------------------------------------------------------
modo_alineacion = 'fases';   % 'fases' (por tramos) | 'global' (lineal unica)
modo_variantes  = 'por_variante'; % 'por_variante': una figura por secuencia de fases
                                 % 'todas': todas las repeticiones en una sola figura
min_rep_variante = 2;        % repeticiones minimas para caracterizar una variante
mostrar_figuras  = 'auto';   % 'auto': se muestran si es UNA sena; con varias solo se guardan (true / false fuerzan)
N_normalizado   = 101;       % puntos del eje normalizado (0,1,...,100 %)
max_hueco_s     = 0.25;      % huecos NaN de hasta este tiempo (s) se interpolan
brazo_analisis  = 'auto';    % brazo que se grafica en TODAS las repeticiones de una sena:
                             % 'auto' = el dominante en la mayoria de sus repeticiones | 'der' | 'izq'
guardar_salidas = true;      % guarda PNG + Excel de resumen en la carpeta

% Colores de las franjas segun el TIPO de fase (columna 'Tipo' de la hoja Fases)
colores_tipo.Sostenimiento = [0.86 0.86 0.86];   % gris
colores_tipo.Movimiento    = [1.00 0.85 0.60];   % naranja
color_tipo_otro            = [0.80 0.90 1.00];   % azul (tipo no reconocido)
alpha_fases = 0.35;   % fondo de fases en las graficas
alpha_std   = 0.45;

% Articulaciones de los dedos promediadas para la "Flexion Dedos"
% (se antepone 'ang_manoD_' o 'ang_manoI_' segun el brazo dominante)
dedos_flex = {'indice_mcp', 'indice_pip', 'medio_mcp', ...
              'medio_pip',  'anular_mcp', 'anular_pip'};

%% 1. SELECCION DE LA CARPETA Y EMPAREJAMIENTO DE ARCHIVOS ---------------
carpeta = uigetdir(pwd, 'Selecciona la carpeta de una sena o la carpeta resultados completa');
if isequal(carpeta, 0)
    disp('Operacion cancelada por el usuario.');
    return;
end

pares = emparejar_archivos(carpeta);
if isempty(pares)
    error(['No se encontraron carpetas con angulos.xlsx + fases.xlsx dentro de "%s".\n' ...
           'Selecciona la carpeta de una sena o la carpeta resultados.'], carpeta);
end

fprintf('Repeticiones encontradas (%d):\n', numel(pares));
for i = 1:numel(pares)
    fprintf('  %2d) %s\n', i, ruta_relativa(fileparts(pares(i).angulos), carpeta));
end

%% 2. LECTURA DE CADA REPETICION ------------------------------------------
reps = {};
for i = 1:numel(pares)
    try
        fa = leer_fases(pares(i).fases);
        an = leer_angulos(pares(i).angulos, dedos_flex, max_hueco_s);
    catch ME
        warning('Se omite la repeticion %d (%s): %s', i, ...
                pares(i).nombre, ME.message);
        continue;
    end

    % El tramo de fases debe caer dentro del tiempo registrado en angulos
    tol = 0.05;  % s
    if fa.lim(1) < an.t(1) - tol || fa.lim(end) > an.t(end) + tol
        warning(['Repeticion %d: las fases (%.2f-%.2f s) salen del rango de ' ...
                 'angulos (%.2f-%.2f s). Se recortan al rango disponible.'], ...
                 i, fa.lim(1), fa.lim(end), an.t(1), an.t(end));
    end

    r = fa;
    r.brazo_detectado = fa.brazo;   % el que eligio Python en ESTA repeticion
    r.t     = an.t;
    r.lados = an;                   % angulos de los dos brazos; el que se usa se fija por sena
    r.nombre_archivo = pares(i).nombre;   % nombre de la carpeta de la repeticion
    r.carpeta_sena   = fileparts(fileparts(pares(i).angulos));   % .../resultados/SENA
    reps{end+1} = r; %#ok<SAGROW>
end

num_rep = numel(reps);
if num_rep == 0
    error('Ninguna repeticion pudo leerse correctamente.');
end
if num_rep < 2
    warning('Solo hay %d repeticion valida: la desviacion estandar sera 0.', num_rep);
end
fprintf('Se van a procesar %d repeticiones.\n', num_rep);

%% 3. CARACTERIZACION POR SENA Y POR VARIANTE -----------------------------
% Cada sena se identifica por la carpeta que contiene sus repeticiones
% (resultados/SENA/Persona_repN). Dentro de cada sena, las repeticiones con
% distinta secuencia de fases (p. ej. 7 vs 9 fases porque en unas la persona
% hace una pausa a mitad del gesto) no se pueden alinear fase a fase entre si:
% en modo 'por_variante' se agrupan por su secuencia (S = sostenimiento,
% M = movimiento) y se caracteriza cada grupo por separado.
carpetas_sena = cellfun(@(r) r.carpeta_sena, reps, 'UniformOutput', false);
[lista_senas, ~, idx_sena] = unique(carpetas_sena);
num_senas = numel(lista_senas);
fprintf('Senas encontradas: %d\n', num_senas);

if ischar(mostrar_figuras)
    mostrar_figuras = (num_senas == 1);
end
cfg = struct('modo_alineacion', modo_alineacion, 'N_normalizado', N_normalizado, ...
             'guardar_salidas', guardar_salidas, 'colores_tipo', colores_tipo, ...
             'color_tipo_otro', color_tipo_otro, 'alpha_fases', alpha_fases, ...
             'alpha_std', alpha_std, 'mostrar_figuras', mostrar_figuras);

resumen_general = cell(0, 8);   % filas: sena, variante, secuencia, n, duracion media, duracion std, repeticiones, estado
for s_i = 1:num_senas
    reps_sena    = reps(idx_sena == s_i);
    carpeta_sena = lista_senas{s_i};
    [~, nombre_carpeta] = fileparts(carpeta_sena);

    % Nombre de la sena: el mas frecuente en las hojas Info (si no, el de la carpeta)
    senas = cellfun(@(r) r.sena, reps_sena, 'UniformOutput', false);
    [senas_unicas, ~, idx_s] = unique(senas);
    nombre_sena = senas_unicas{mode(idx_s)};
    if strcmp(nombre_sena, 'SENA'), nombre_sena = nombre_carpeta; end
    if numel(senas_unicas) > 1
        warning('En "%s" hay repeticiones de senas distintas: %s. Se usa "%s".', ...
                nombre_carpeta, strjoin(senas_unicas, ', '), nombre_sena);
    end
    % Mismo brazo para todas las repeticiones de la sena. En senas simetricas a
    % dos manos (p. ej. "las manos se juntan") Python puede elegir un brazo
    % dominante distinto en cada repeticion; promediar el codo derecho de unas
    % con el izquierdo de otras mezclaria dos articulaciones diferentes.
    detectados = cellfun(@(r) r.brazo_detectado, reps_sena, 'UniformOutput', false);
    n_der = sum(strcmp(detectados, 'der'));
    n_izq = sum(strcmp(detectados, 'izq'));
    if strcmp(brazo_analisis, 'auto')
        if n_izq > n_der, lado_sena = 'izq'; else, lado_sena = 'der'; end
    else
        lado_sena = brazo_analisis;
    end
    for i = 1:numel(reps_sena)
        reps_sena{i}.brazo  = lado_sena;
        reps_sena{i}.codo   = reps_sena{i}.lados.(lado_sena).codo;
        reps_sena{i}.hombro = reps_sena{i}.lados.(lado_sena).hombro;
        reps_sena{i}.flex   = reps_sena{i}.lados.(lado_sena).flex;
    end
    if n_der > 0 && n_izq > 0
        distintas = cellfun(@(r) r.nombre_archivo, reps_sena(~strcmp(detectados, lado_sena)), ...
                            'UniformOutput', false);
        fprintf(['  Brazo dominante detectado: der en %d, izq en %d repeticiones -> se grafica ' ...
                 'el brazo %s en todas.\n  Repeticiones con el otro brazo como dominante: %s\n'], ...
                n_der, n_izq, upper(lado_sena), strjoin(distintas, ', '));
    end

    fprintf('\n=================================================================\n');
    fprintf('[%d/%d] %s  (%d repeticiones)\n', s_i, num_senas, nombre_sena, numel(reps_sena));
    fprintf('=================================================================\n');
    patrones = cellfun(@(r) patron_fases(r.tipo), reps_sena, 'UniformOutput', false);
    for i = 1:numel(reps_sena)
        fprintf('  %-16s %2d fases (%s)\n', reps_sena{i}.nombre_archivo, ...
                numel(reps_sena{i}.tipo), patrones{i});
    end

    try
        if strcmp(modo_variantes, 'todas')
            info = caracterizar_grupo(reps_sena, nombre_sena, carpeta_sena, cfg, '', '');
            resumen_general(end+1, :) = {nombre_sena, 'todas', '-', numel(reps_sena), ...
                info.dur_media, info.dur_std, info.nombres, 'caracterizada'}; %#ok<SAGROW>
            continue;
        end
        [pat_unicos, ~, idx_p] = unique(patrones);
        conteo = accumarray(idx_p(:), 1);
        [~, orden] = sort(conteo, 'descend');
        fprintf('  Variantes:\n');
        for g = orden(:)'
            miembros = find(idx_p == g);
            nombres  = strjoin(cellfun(@(r) r.nombre_archivo, reps_sena(miembros), ...
                                       'UniformOutput', false), ', ');
            n_fases  = numel(reps_sena{miembros(1)}.tipo);
            variante = sprintf('%d fases', n_fases);
            if numel(miembros) < min_rep_variante
                fprintf('    %d fases (%s): %d repeticion(es) -> se omite (minimo %d): %s\n', ...
                        n_fases, pat_unicos{g}, numel(miembros), min_rep_variante, nombres);
                resumen_general(end+1, :) = {nombre_sena, variante, pat_unicos{g}, ...
                    numel(miembros), NaN, NaN, nombres, 'omitida (pocas repeticiones)'}; %#ok<SAGROW>
                continue;
            end
            fprintf('    %d fases (%s): %d repeticiones -> %s\n', ...
                    n_fases, pat_unicos{g}, numel(miembros), nombres);
            etiqueta = sprintf(' - variante %d fases (n = %d)', n_fases, numel(miembros));
            sufijo   = sprintf('_%dfases', n_fases);
            if sum(cellfun(@(p) numel(p), pat_unicos) == numel(pat_unicos{g})) > 1
                sufijo = [sufijo '_' strrep(pat_unicos{g}, '-', '')];   % mismo n, distinta secuencia
            end
            info = caracterizar_grupo(reps_sena(miembros), nombre_sena, carpeta_sena, cfg, ...
                                      etiqueta, sufijo);
            resumen_general(end+1, :) = {nombre_sena, variante, pat_unicos{g}, numel(miembros), ...
                info.dur_media, info.dur_std, nombres, 'caracterizada'}; %#ok<SAGROW>
        end
    catch ME
        warning('No se pudo caracterizar "%s": %s', nombre_sena, ME.message);
        resumen_general(end+1, :) = {nombre_sena, '-', '-', numel(reps_sena), NaN, NaN, ...
            '', ['error: ' ME.message]}; %#ok<SAGROW>
    end
end

%% 4. RESUMEN DE TODAS LAS SENAS --------------------------------------------
if num_senas > 1 && guardar_salidas && ~isempty(resumen_general)
    T_general = cell2table(resumen_general, 'VariableNames', {'Sena', 'Variante', ...
        'Secuencia_fases', 'N_repeticiones', 'Duracion_media_s', 'Duracion_std_s', ...
        'Repeticiones', 'Estado'});
    ruta_general = fullfile(carpeta, 'resumen_todas_las_senas.xlsx');
    try
        if isfile(ruta_general), delete(ruta_general); end
        writetable(T_general, ruta_general);
        fprintf('\nResumen de todas las senas: %s\n', ruta_general);
    catch ME
        warning('No se pudo guardar el resumen general (esta abierto?): %s', ME.message);
    end
end
fprintf('\nListo: %d sena(s) procesada(s).\n', num_senas);

% =========================================================================
%                          FUNCIONES LOCALES
% =========================================================================
function info = caracterizar_grupo(reps, nombre_sena, carpeta, cfg, etiqueta, sufijo)
% Caracteriza un grupo de repeticiones: alineacion temporal, media +/- sigma,
% figura con barra de fases y Excel de resumen.
%   etiqueta: texto que se agrega al titulo (ej. ' - variante 7 fases')
%   sufijo  : texto que se agrega al nombre de los archivos (ej. '_7fases')
    modo_alineacion = cfg.modo_alineacion;
    N_normalizado   = cfg.N_normalizado;
    guardar_salidas = cfg.guardar_salidas;
    colores_tipo    = cfg.colores_tipo;
    color_tipo_otro = cfg.color_tipo_otro;
    alpha_fases     = cfg.alpha_fases;
    alpha_std       = cfg.alpha_std;
    num_rep = numel(reps);
    brazos  = cellfun(@(r) r.brazo, reps, 'UniformOutput', false);

    % --- G1. ESTRUCTURA DE FASES COMUN -------------------------------------------
    num_fases = cellfun(@(r) numel(r.lim) - 1, reps);
    K_ref     = mode(num_fases);
    idx_ref   = find(num_fases == K_ref);

    if any(num_fases ~= K_ref)
        for i = find(num_fases ~= K_ref)
            warning('"%s" tiene %d fases (la mayoria tiene %d).', ...
                    reps{i}.nombre_archivo, num_fases(i), K_ref);
        end
        if strcmp(modo_alineacion, 'fases')
            warning(['No todas las repeticiones tienen el mismo numero de fases: ' ...
                     'se usa alineacion ''global''. Los limites de fase se ' ...
                     'calculan solo con las repeticiones de %d fases.'], K_ref);
            modo_alineacion = 'global';
        end
    end

    % Nombres, tipos y descripciones de referencia
    fases_ref = reps{idx_ref(1)}.fase;
    tipos_ref = reps{idx_ref(1)}.tipo;
    desc_ref  = reps{idx_ref(1)}.desc;
    for i = idx_ref(2:end)
        if ~isequal(reps{i}.tipo, tipos_ref)
            warning('"%s": la secuencia de tipos de fase no coincide con la de referencia.', ...
                    reps{i}.nombre_archivo);
        end
    end

    % Limites en % del tramo activo y duraciones (s) de cada repeticion
    n_ref    = numel(idx_ref);
    lim_pct  = zeros(n_ref, K_ref + 1);
    dur_fase = zeros(n_ref, K_ref);
    for j = 1:n_ref
        b = reps{idx_ref(j)}.lim;
        lim_pct(j, :)  = 100 * (b - b(1)) / (b(end) - b(1));
        dur_fase(j, :) = diff(b);
    end
    lim_pct_media = mean(lim_pct, 1);
    lim_pct_std   = std(lim_pct, 0, 1);
    lim_pct_media([1 end]) = [0 100];   % exactos (evita errores de redondeo)

    % --- G2. NORMALIZACION TEMPORAL DE CADA REPETICION ---------------------------
    t_eje = linspace(0, 100, N_normalizado);

    matriz_codo   = nan(num_rep, N_normalizado);
    matriz_hombro = nan(num_rep, N_normalizado);
    matriz_flex   = nan(num_rep, N_normalizado);

    for i = 1:num_rep
        r = reps{i};
        b = r.lim;

        if strcmp(modo_alineacion, 'fases')
            % Por tramos: el % promedio de cada limite se mapea al tiempo real
            % del mismo limite en esta repeticion (interpolacion lineal a trozos)
            t_obj = interp1(lim_pct_media, b, t_eje, 'linear');
        else
            % Lineal unica sobre el tramo activo
            t_obj = b(1) + (b(end) - b(1)) * t_eje / 100;
        end
        t_obj = min(max(t_obj, r.t(1)), r.t(end));

        matriz_codo(i, :)   = interp1(r.t, r.codo,   t_obj, 'linear');
        matriz_hombro(i, :) = interp1(r.t, r.hombro, t_obj, 'linear');
        matriz_flex(i, :)   = interp1(r.t, r.flex,   t_obj, 'linear');
    end

    % --- G3. MEDIA Y DESVIACION ESTANDAR -----------------------------------------
    media_codo   = mean(matriz_codo,   1, 'omitnan');  std_codo   = std(matriz_codo,   0, 1, 'omitnan');
    media_hombro = mean(matriz_hombro, 1, 'omitnan');  std_hombro = std(matriz_hombro, 0, 1, 'omitnan');
    media_flex   = mean(matriz_flex,   1, 'omitnan');  std_flex   = std(matriz_flex,   0, 1, 'omitnan');

    n_validas_flex = sum(~isnan(matriz_flex), 1);
    if any(n_validas_flex < num_rep)
        fprintf(['Aviso: en %d de %d puntos la flexion de dedos se calculo con menos ' ...
                 'de %d repeticiones (mano no detectada).\n'], ...
                 sum(n_validas_flex < num_rep), N_normalizado, num_rep);
    end

    % --- G4. GRAFICAS --------------------------------------------------------------
    % Distribucion vertical: barra de fases arriba + 3 graficas articulares.
    %   - Barra de fases: un bloque por fase ("Fase 1", "Fase 2", ...) coloreado
    %     por tipo (sostenimiento / movimiento). La barrita negra en cada limite
    %     indica la variabilidad (+/- sigma, en % del gesto) del momento en que
    %     empieza esa fase entre las repeticiones.
    %   - Graficas: fondo suave con los mismos colores de fase, lineas grises
    %     discontinuas en los limites, banda +/- sigma y media de cada articulacion.
    visible = 'on';
    if ~cfg.mostrar_figuras, visible = 'off'; end
    fig = figure('Name', ['Caracterizacion Biomecanica - ' nombre_sena ' ' etiqueta], ...
                 'Position', [100, 40, 1000, 1000], 'Color', 'w', 'Visible', visible);
    tl = tiledlayout(fig, 11, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    tl.OuterPosition = [0 0.07 1 0.93];   % deja espacio abajo para las descripciones

    title(tl, {sprintf('Caracterizacion de Sena: %s%s', nombre_sena, etiqueta), ...
               sprintf('Media \\pm \\sigma de %d repeticiones  |  brazo %s  |  alineacion por %s', ...
                       num_rep, upper(brazos{1}), modo_alineacion)}, ...
          'FontSize', 13, 'FontWeight', 'bold');

    lim_internos = lim_pct_media(2:end-1);   % limites entre fases (sin 0 ni 100)
    std_internos = lim_pct_std(2:end-1);

    % ---------- 6.1 Barra de fases ----------
    ax_fases = nexttile(tl, [2 1]);
    hold(ax_fases, 'on');
    for f = 1:K_ref
        x0  = lim_pct_media(f);
        x1  = lim_pct_media(f + 1);
        col = color_de_tipo(tipos_ref{f}, colores_tipo, color_tipo_otro);
        patch(ax_fases, [x0 x1 x1 x0], [0 0 1 1], col, ...
              'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 0.8);

        ancho = x1 - x0;
        num_f = regexp(fases_ref{f}, '\d+', 'match', 'once');
        if isempty(num_f), num_f = num2str(f); end
        if ancho >= 9
            etiqueta_f = sprintf('Fase %s', num_f);
        else
            etiqueta_f = sprintf('F%s', num_f);    % fase muy corta: etiqueta compacta
        end
        text(ax_fases, (x0 + x1) / 2, 0.68, etiqueta_f, ...
             'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
        if ancho >= 9
            text(ax_fases, (x0 + x1) / 2, 0.36, sprintf('%.0f%%', ancho), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.35 0.35 0.35]);
        end
    end
    % Variabilidad de cada limite (+/- sigma entre repeticiones)
    for f = 1:numel(lim_internos)
        m = lim_internos(f);  s = std_internos(f);
        if s > 0
            plot(ax_fases, [m - s, m + s], [0.12 0.12], 'k-', 'LineWidth', 1.5);
            plot(ax_fases, [m - s, m - s], [0.05 0.19], 'k-', 'LineWidth', 1.5);
            plot(ax_fases, [m + s, m + s], [0.05 0.19], 'k-', 'LineWidth', 1.5);
        end
    end
    ylim(ax_fases, [0 1]);
    set(ax_fases, 'YTick', [], 'XTickLabel', [], 'Box', 'on', 'Layer', 'top');
    ylabel(ax_fases, 'Fases', 'FontWeight', 'bold');

    % ---------- 6.2 Graficas articulares ----------
    datos = {media_codo,   std_codo,   [0 0 1],   sprintf('Angulo Codo %s (\\circ)',   upper(brazos{1}));
             media_hombro, std_hombro, [1 0 0],   sprintf('Angulo Hombro %s (\\circ)', upper(brazos{1}));
             media_flex,   std_flex,   [0 0.6 0], 'Flexion Dedos (\circ)'};

    axes_list = gobjects(3, 1);
    for k = 1:3
        ax = nexttile(tl, [3 1]);
        axes_list(k) = ax;
        hold(ax, 'on');

        media = datos{k, 1};  sd = datos{k, 2};  color = datos{k, 3};
        arriba = media + sd;  abajo = media - sd;
        y_min = min(abajo, [], 'omitnan');  y_max = max(arriba, [], 'omitnan');
        if isempty(y_min) || isnan(y_min), y_min = 0; y_max = 1; end
        margen = 0.08 * (y_max - y_min + eps);
        ylims  = [y_min - margen, y_max + margen];

        % Fondo de fases (suave, solo como guia)
        h_tipos = gobjects(0);  tipos_leyenda = {};
        for f = 1:K_ref
            x0  = lim_pct_media(f);  x1 = lim_pct_media(f + 1);
            col = color_de_tipo(tipos_ref{f}, colores_tipo, color_tipo_otro);
            hp  = patch(ax, [x0 x1 x1 x0], [ylims(1) ylims(1) ylims(2) ylims(2)], col, ...
                        'EdgeColor', 'none', 'FaceAlpha', alpha_fases);
            if ~any(strcmp(tipos_leyenda, tipos_ref{f}))
                h_tipos(end+1) = hp;  tipos_leyenda{end+1} = tipos_ref{f}; %#ok<SAGROW>
            end
        end
        % Limites entre fases
        h_lim = gobjects(0);
        for f = 1:numel(lim_internos)
            h_lim = plot(ax, [lim_internos(f) lim_internos(f)], ylims, '--', ...
                         'Color', [0.45 0.45 0.45], 'LineWidth', 1);
        end

        % Banda +/- sigma y media
        validos = ~isnan(arriba) & ~isnan(abajo);
        tx = t_eje(validos);
        h_sd  = fill(ax, [tx, fliplr(tx)], [arriba(validos), fliplr(abajo(validos))], color, ...
                     'EdgeColor', 'none', 'FaceAlpha', alpha_std);
        h_med = plot(ax, t_eje, media, 'Color', color, 'LineWidth', 2.2);

        ylabel(ax, datos{k, 4});
        ylim(ax, ylims);
        grid(ax, 'on');  box(ax, 'on');
        set(ax, 'Layer', 'top');
        if k < 3, set(ax, 'XTickLabel', []); end

        if k == 1
            tipos_txt = strcat('Fase de', {' '}, lower(tipos_leyenda));
            h_leg = [h_sd, h_med, h_tipos];
            txt_leg = [{'Desv. estandar (\pm\sigma)', 'Media'}, tipos_txt];
            if ~isempty(h_lim)
                h_leg(end+1) = h_lim;  txt_leg{end+1} = 'Limite entre fases';
            end
            legend(ax, h_leg, txt_leg, 'Location', 'best', 'FontSize', 8);
        end
    end

    xlabel(axes_list(3), 'Progreso Normalizado del Gesto (%)');
    linkaxes([ax_fases; axes_list], 'x');
    xlim(axes_list(3), [0, 100]);

    % ---------- 6.3 Descripcion de las fases (pie de figura) ----------
    desc_txt = cell(1, K_ref);
    for f = 1:K_ref
        desc_txt{f} = sprintf('%s (%s): %s', fases_ref{f}, lower(tipos_ref{f}), desc_ref{f});
    end
    mitad = ceil(K_ref / 2);
    nota = sprintf(['Barra negra en la franja de fases: variabilidad (\\pm\\sigma) del ' ...
                    'inicio de cada fase entre repeticiones.']);
    annotation(fig, 'textbox', [0.03 0.005 0.94 0.065], ...
               'String', {strjoin(desc_txt(1:mitad), '   |   '), ...
                          strjoin(desc_txt(mitad+1:end), '   |   '), nota}, ...
               'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
               'VerticalAlignment', 'middle', 'FontSize', 8.5, 'Interpreter', 'tex');

    % --- G5. RESUMEN EN CONSOLA Y ARCHIVOS DE SALIDA -----------------------------
    fprintf('\nFases de "%s"%s (%d repeticiones con %d fases):\n', nombre_sena, etiqueta, n_ref, K_ref);
    fprintf('  %-4s %-14s %-28s %16s %22s\n', 'Fase', 'Tipo', 'Descripcion', ...
            'Duracion (s)', 'Inicio (% gesto)');
    for f = 1:K_ref
        fprintf('  %-4s %-14s %-28s %7.2f +/- %5.2f %10.1f +/- %5.1f\n', ...
                fases_ref{f}, tipos_ref{f}, desc_ref{f}, ...
                mean(dur_fase(:, f)), std(dur_fase(:, f), 0), ...
                lim_pct_media(f), lim_pct_std(f));
    end
    dur_total = sum(dur_fase, 2);
    info = struct('dur_media', mean(dur_total), 'dur_std', std(dur_total, 0), ...
                  'nombres', strjoin(cellfun(@(r) r.nombre_archivo, reps(idx_ref), ...
                                             'UniformOutput', false), ', '));
    fprintf('  Duracion total del tramo activo: %.2f +/- %.2f s\n\n', ...
            mean(dur_total), std(dur_total, 0));

    if guardar_salidas
        nombre_base = regexprep(nombre_sena, '[^\w\-]', '_');

        % Figura
        ruta_png = fullfile(carpeta, ['caracterizacion_' nombre_base sufijo '.png']);
        try
            if exist('exportgraphics', 'file')
                exportgraphics(fig, ruta_png, 'Resolution', 200);
            else
                saveas(fig, ruta_png);
            end
            fprintf('Figura guardada en: %s\n', ruta_png);
        catch ME
            warning('No se pudo guardar la figura: %s', ME.message);
        end

        % Excel de resumen (3 hojas)
        ruta_xlsx = fullfile(carpeta, ['resumen_' nombre_base sufijo '.xlsx']);
        try
            T_fases = table(fases_ref(:), tipos_ref(:), desc_ref(:), ...
                mean(dur_fase, 1)', std(dur_fase, 0, 1)', ...
                lim_pct_media(1:end-1)', lim_pct_std(1:end-1)', ...
                lim_pct_media(2:end)',   lim_pct_std(2:end)', ...
                'VariableNames', {'Fase', 'Tipo', 'Descripcion', ...
                                  'Duracion_media_s', 'Duracion_std_s', ...
                                  'Inicio_pct_media', 'Inicio_pct_std', ...
                                  'Fin_pct_media', 'Fin_pct_std'});

            nombres_rep = cellfun(@(r) r.nombre_archivo, reps(idx_ref), 'UniformOutput', false);
            T_rep = [table(nombres_rep(:), dur_total, 'VariableNames', {'Repeticion', 'Duracion_total_s'}), ...
                     array2table(dur_fase, 'VariableNames', strcat(fases_ref, '_s'))];

            T_curvas = table(t_eje', media_codo', std_codo', media_hombro', std_hombro', ...
                             media_flex', std_flex', n_validas_flex', ...
                'VariableNames', {'Progreso_pct', 'Codo_media_grados', 'Codo_std_grados', ...
                                  'Hombro_media_grados', 'Hombro_std_grados', ...
                                  'FlexDedos_media_grados', 'FlexDedos_std_grados', ...
                                  'FlexDedos_n_repeticiones'});

            if isfile(ruta_xlsx), delete(ruta_xlsx); end
            writetable(T_fases,  ruta_xlsx, 'Sheet', 'Fases');
            writetable(T_rep,    ruta_xlsx, 'Sheet', 'Repeticiones');
            writetable(T_curvas, ruta_xlsx, 'Sheet', 'Curvas');
            fprintf('Resumen guardado en: %s\n', ruta_xlsx);
        catch ME
            warning('No se pudo guardar el Excel de resumen (esta abierto?): %s', ME.message);
        end
    end
    if ~cfg.mostrar_figuras
        close(fig);   % con muchas senas no se dejan decenas de ventanas abiertas
    end
end

function p = patron_fases(tipos)
% 'S-M-S-...' a partir de los tipos de fase (Sostenimiento / Movimiento)
    p = strjoin(cellfun(@(x) upper(x(1:min(1, end))), tipos, 'UniformOutput', false), '-');
end

function pares = emparejar_archivos(carpeta)
% Busca, en la carpeta seleccionada y en todas sus subcarpetas, las carpetas
% de repeticion que contienen EXACTAMENTE 'angulos.xlsx' y 'fases.xlsx'.
% El resto de archivos (dedos_*.xlsx, landmarks.xlsx, CSV, PNG, ...) se ignora.
%   ESTACIONDEPOLICIA/
%       Todos_rep1/  angulos.xlsx  fases.xlsx  ...
%       Todos_rep2/  angulos.xlsx  fases.xlsx  ...
    lista = dir(fullfile(carpeta, '**', 'angulos.xlsx'));
    pares = struct('angulos', {}, 'fases', {}, 'nombre', {});

    for j = 1:numel(lista)
        dir_rep    = lista(j).folder;
        ruta_ang   = fullfile(dir_rep, lista(j).name);
        ruta_fases = fullfile(dir_rep, 'fases.xlsx');
        [~, nombre_rep] = fileparts(dir_rep);   % ej. 'Todos_rep3'

        if ~isfile(ruta_fases)
            warning('"%s" tiene angulos.xlsx pero no fases.xlsx: se omite.', nombre_rep);
            continue;
        end
        pares(end+1) = struct('angulos', ruta_ang, 'fases', ruta_fases, ...
                              'nombre', nombre_rep); %#ok<AGROW>
    end

    % Orden natural: rep1, rep2, ..., rep10 (no rep1, rep10, rep11, rep2)
    if ~isempty(pares)
        num = cellfun(@numero_final, {pares.nombre});
        [~, orden] = sortrows([num(:), (1:numel(num))']);
        pares = pares(orden);
    end
end

function n = numero_final(texto)
% Ultimo numero que aparece en el texto (Inf si no hay ninguno)
    tok = regexp(texto, '\d+', 'match');
    if isempty(tok), n = Inf; else, n = str2double(tok{end}); end
end

function fa = leer_fases(ruta)
% Lee la hoja 'Fases' (limites, nombre, tipo, descripcion) y la hoja 'Info'
% (sena y brazo dominante). Los tiempos estan en segundos de video.
    opts = detectImportOptions(ruta, 'Sheet', 'Fases', 'VariableNamingRule', 'preserve');
    T = readtable(ruta, opts);

    % Tiempos en el VIDEO (mismo eje que tiempo_ms de angulos.xlsx).
    %  - fases.xlsx nuevos: columnas 't inicio video (s)' / 't fin video (s)'.
    %  - fases.xlsx viejos: 't inicio (s)' / 't fin (s)' estan medidos desde el
    %    inicio del TRAMO ACTIVO; se les suma el 'Inicio del gesto' (hoja Eventos).
    if tiene_columna(T, 't inicio video')
        t_ini = double(T.(buscar_columna(T, 't inicio video')));
        t_fin = double(T.(buscar_columna(T, 't fin video')));
    else
        t_ini = double(T.(buscar_columna(T, 't inicio')));
        t_fin = double(T.(buscar_columna(T, 't fin')));
        inicio_gesto = leer_inicio_gesto(ruta);
        t_ini = t_ini + inicio_gesto;
        t_fin = t_fin + inicio_gesto;
    end
    if isempty(t_ini) || any(isnan([t_ini; t_fin]))
        error('La hoja Fases no tiene tiempos validos.');
    end

    lim = [t_ini(:); t_fin(end)]';
    if any(diff(lim) <= 0)
        error('Los limites de fase no son estrictamente crecientes.');
    end
    fa.lim  = lim;
    fa.fase = texto_celda(T.(buscar_columna(T, 'fase')));
    fa.tipo = texto_celda(T.(buscar_columna(T, 'tipo')));
    fa.desc = texto_celda(T.(buscar_columna(T, 'descripcion')));

    fa.sena  = 'SENA';
    fa.brazo = 'der';
    try
        Ti = readtable(ruta, 'Sheet', 'Info', 'VariableNamingRule', 'preserve');
        fa.sena  = char(string(Ti.(buscar_columna(Ti, 'sena'))(1)));
        fa.brazo = lower(strtrim(char(string(Ti.(buscar_columna(Ti, 'brazo'))(1)))));
    catch
        warning('No se pudo leer la hoja Info de %s; se asume brazo derecho.', ruta);
    end
    if ~ismember(fa.brazo, {'der', 'izq'})
        warning('Brazo dominante "%s" no reconocido en %s; se usa ''der''.', fa.brazo, ruta);
        fa.brazo = 'der';
    end
end

function an = leer_angulos(ruta, dedos_flex, max_hueco_s)
% Lee la hoja 'Angulos' UNA vez y extrae, para los DOS brazos, codo, hombro y
% flexion media de dedos:  an.t, an.der.codo/hombro/flex, an.izq.codo/hombro/flex.
% Rellena huecos NaN cortos (<= max_hueco_s) por interpolacion lineal usando
% las marcas de tiempo reales. Si faltan columnas de un brazo quedan en NaN.
    opts = detectImportOptions(ruta, 'Sheet', 'Angulos');
    T = readtable(ruta, opts);
    nombres = T.Properties.VariableNames;
    if ~ismember('tiempo_ms', nombres)
        error('Falta la columna tiempo_ms.');
    end

    % Tiempos ordenados y sin duplicados (requisito de interp1)
    t_bruto = double(T.tiempo_ms) / 1000;
    validos = find(~isnan(t_bruto));
    [t, iu] = unique(t_bruto(validos));
    filas   = validos(iu);
    if numel(t) < 2
        error('Menos de 2 muestras validas de tiempo.');
    end
    an.t = t;

    lados = {'der', 'manoD'; 'izq', 'manoI'};
    hay_alguno = false;
    for k = 1:2
        lado = lados{k, 1};  mano = lados{k, 2};
        col_codo   = ['ang_codo_' lado];
        col_hombro = ['ang_hombro_' lado];
        cols_flex  = intersect(strcat('ang_', mano, '_', dedos_flex), nombres);

        codo = nan(size(t));  hombro = nan(size(t));  flex = nan(size(t));
        if ismember(col_codo, nombres),   codo   = double(T.(col_codo)(filas));     hay_alguno = true; end
        if ismember(col_hombro, nombres), hombro = double(T.(col_hombro)(filas)); end
        if ~isempty(cols_flex)
            flex = mean(table2array(T(filas, cols_flex)), 2, 'omitnan');
        end
        an.(lado).codo   = rellenar_huecos(codo,   t, max_hueco_s);
        an.(lado).hombro = rellenar_huecos(hombro, t, max_hueco_s);
        an.(lado).flex   = rellenar_huecos(flex,   t, max_hueco_s);
    end
    if ~hay_alguno
        error('No hay columnas ang_codo_der / ang_codo_izq.');
    end
end

function x = rellenar_huecos(x, t, max_hueco_s)
    if any(isnan(x)) && sum(~isnan(x)) >= 2
        x = fillmissing(x, 'linear', 'SamplePoints', t, 'MaxGap', max_hueco_s);
    end
end

function t0 = leer_inicio_gesto(ruta)
% 'Inicio del gesto' (s, tiempo del video) de la hoja Eventos. Si no existe
% se asume 0 con un aviso (los tiempos de fase quedarian relativos al tramo).
    t0 = 0;
    try
        E = readtable(ruta, 'Sheet', 'Eventos', 'VariableNamingRule', 'preserve');
        eventos = strtrim(string(E.(buscar_columna(E, 'evento'))));
        k = find(strcmpi(eventos, 'Inicio del gesto'), 1);
        valor = double(E.(buscar_columna(E, 't video'))(k));
        if ~isempty(k) && isfinite(valor)
            t0 = valor;
            return;
        end
    catch
    end
    warning('No se encontro "Inicio del gesto" en la hoja Eventos de %s; se asume 0 s.', ruta);
end

function si = tiene_columna(T, patron)
    si = any(startsWith(lower(strtrim(T.Properties.VariableNames)), lower(patron)));
end

function nombre = buscar_columna(T, patron)
% Devuelve el nombre de la primera columna cuyo encabezado empieza por
% 'patron' (sin distinguir mayusculas).
    nombres = T.Properties.VariableNames;
    k = find(startsWith(lower(strtrim(nombres)), lower(patron)), 1);
    if isempty(k)
        error('No se encontro la columna "%s".', patron);
    end
    nombre = nombres{k};
end

function c = texto_celda(v)
    s = strtrim(string(v));
    s(ismissing(s)) = "";
    c = cellstr(s)';
end

function col = color_de_tipo(tipo, colores_tipo, color_otro)
    campo = matlab.lang.makeValidName(tipo);
    if isfield(colores_tipo, campo)
        col = colores_tipo.(campo);
    else
        col = color_otro;
    end
end

function r = ruta_relativa(ruta, carpeta)
    r = strrep(ruta, [carpeta filesep], '');
end



