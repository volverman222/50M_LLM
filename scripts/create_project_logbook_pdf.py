from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, Paragraph, Preformatted, Spacer, Table, TableStyle,
)


OUT = Path("output/pdf/cuaderno_bitacora_proyecto_llm.pdf")
V41_GRAPH = Path("assets/v4_1_train_validation_loss.png")
V42_LR_GRAPH = Path("assets/v4_2_cosine_lr_schedule.png")
V42_TRAIN_GRAPH = Path("assets/v4_2_train_loss_vs_v4_1.png")
V42_VAL_GRAPH = Path("assets/v4_2_validation_vs_v4_1.png")
V43_VAL_GRAPH = Path("assets/v4_3_xavier_validation_vs_v4_2.png")
V43_TRAIN_GRAPH = Path("assets/v4_3_xavier_train_vs_v4_2.png")
V433_VAL_GRAPH = Path("assets/v4_3_3_swiglu_validation_vs_xavier.png")
V44_VAL_GRAPH = Path("assets/v4_4_more_depth_validation_vs_swiglu.png")
V45_VAL_GRAPH = Path("assets/v4_5_looped_transformer_validation_vs_swiglu.png")
V47_VAL_GRAPH = Path("assets/v4_7_rope_validation_vs_looped.png")
V48_VAL_GRAPH = Path("assets/v4_8_small_vocab_validation_vs_rope.png")
NAVY = colors.HexColor("#13233F")
BLUE = colors.HexColor("#2563EB")
SKY = colors.HexColor("#EAF2FF")
MINT = colors.HexColor("#E8F7F2")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#CBD5E1")


def line_table(rows, widths=None, shade=False, heights=None):
    table = Table(rows, colWidths=widths, rowHeights=heights, repeatRows=1 if len(rows) > 1 else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if shade:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    table.setStyle(TableStyle(style))
    return table


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=28, leading=34, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=23, textColor=NAVY, spaceBefore=3, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.7, leading=10, textColor=MUTED))
    styles.add(ParagraphStyle(name="CodeCustom", fontName="Courier", fontSize=6.5, leading=8.2, textColor=INK))
    h1, h2, body, small = (styles[x] for x in ("H1Custom", "H2Custom", "BodyCustom", "Small"))

    def p(text, style=body): return Paragraph(text, style)
    def section(title, text=None):
        story.append(p(title, h2))
        if text: story.append(p(text))
    def blank_rows(n, height=22):
        return [[" "] for _ in range(n)]
    def code_card(label, source_path, code):
        story.append(p(label, h2))
        story.append(p(f"Fuente: <font name='Courier'>{source_path}</font>", small))
        snippet = Table([[Preformatted(code.strip(), styles["CodeCustom"])]], colWidths=[17*cm])
        snippet.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), .55, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(snippet)
        story.append(Spacer(1, .18*cm))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2, .92*cm, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(str(OUT), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=1.7*cm, bottomMargin=1.8*cm)
    doc.addPageTemplates([__import__("reportlab.platypus", fromlist=["PageTemplate"]).PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")], onPage=footer)])
    story = []

    story += [Spacer(1, 3.3*cm), p("CUADERNO DE BITACORA", styles["TitleCustom"]), p("Proyecto de entrenamiento y evaluacion de un modelo de lenguaje", ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=13, leading=18, textColor=BLUE)), Spacer(1, .8*cm)]
    story.append(Table([[p("<b>Proyecto</b><br/><br/>50M_LLM", body), p("<b>Periodo</b><br/><br/>________________________", body)], [p("<b>Responsable</b><br/><br/>________________________", body), p("<b>Version del cuaderno</b><br/><br/>v0.1", body)]], colWidths=[8.5*cm, 8.5*cm], rowHeights=[3*cm, 3*cm], style=[("BACKGROUND", (0,0), (-1,-1), SKY), ("GRID", (0,0), (-1,-1), .5, LINE), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 12), ("TOPPADDING", (0,0), (-1,-1), 11)]))
    story += [Spacer(1, 1.2*cm), p("Registra decisiones, evidencias y aprendizajes mientras trabajas. Al final, cada entrada se convierte en material directo para el informe tecnico.", ParagraphStyle("quote", parent=body, alignment=TA_CENTER, textColor=MUTED, fontSize=10.5, leading=15)), PageBreak()]

    story += [p("Como usar este cuaderno", h1), p("Una entrada breve y honesta por cada bloque de trabajo importante vale mas que reconstruir la historia al final. Escribe primero lo que observaste; despues tu interpretacion.")]
    guide = [[p("<b>Cuando</b>", small), p("<b>Que anotar</b>", small), p("<b>Para el informe final</b>", small)],
             [p("Antes de ejecutar", small), p("Hipotesis, configuracion y criterio de exito.", small), p("Metodologia y diseno experimental.", small)],
             [p("Durante", small), p("Cambios, incidencias, coste y observaciones.", small), p("Implementacion y limitaciones.", small)],
             [p("Despues", small), p("Metricas, comparacion con el baseline y conclusion.", small), p("Resultados y discusion.", small)],
             [p("Al decidir", small), p("Alternativas descartadas y por que.", small), p("Justificacion tecnica.", small)]]
    story.append(line_table(guide, [3.1*cm, 6.3*cm, 7.6*cm], True))
    section("Regla practica")
    story.append(Table([[p("1. Fecha y contexto", body), p("2. Evidencia verificable", body), p("3. Decision y siguiente paso", body)]], colWidths=[5.6*cm]*3, style=[("BACKGROUND", (0,0), (-1,-1), MINT), ("GRID", (0,0), (-1,-1), .4, LINE), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Convencion sugerida", "Nombra los experimentos con fecha y objetivo: <b>2026-09-11_lr-sweep_v1</b>. Guarda junto a cada entrada enlaces o rutas a graficas, checkpoints, notebooks y ejecuciones.")
    story.append(PageBreak())

    story += [p("Ficha del proyecto", h1)]
    project = [[p("<b>Problema que resuelve</b><br/><br/><br/>", body), p("<b>Objetivo medible</b><br/><br/><br/>", body)],
               [p("<b>Hipotesis inicial</b><br/><br/><br/>", body), p("<b>Alcance y exclusiones</b><br/><br/><br/>", body)],
               [p("<b>Datos: fuente, licencia, tamano, limpieza</b><br/><br/><br/>", body), p("<b>Metricas principales y baseline</b><br/><br/><br/>", body)],
               [p("<b>Recursos: hardware, tiempo, presupuesto</b><br/><br/><br/>", body), p("<b>Riesgos tecnicos / eticos</b><br/><br/><br/>", body)]]
    story.append(Table(project, colWidths=[8.5*cm, 8.5*cm], rowHeights=[3.0*cm]*4, style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), colors.white), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Configuracion comun hasta V4.3.3", h1), p("Estos valores se han mantenido constantes en las iteraciones documentadas hasta ahora. Constituyen el contexto comun para interpretar las mejoras incrementales.")]
    common_config = [[p("<b>Parametro</b>", small), p("<b>Valor fijo</b>", small), p("<b>Implicacion</b>", small)],
                     [p("SEED", small), p("123", small), p("Reduce variacion de una corrida a otra; una sola seed no mide robustez estadistica.", small)],
                     [p("MAX_LENGTH", small), p("128 tokens", small), p("Longitud maxima de contexto por secuencia.", small)],
                     [p("BATCH_SIZE", small), p("2", small), p("Tamano de lote por actualizacion antes de cualquier acumulacion no registrada.", small)],
                     [p("MAX_TOKENS", small), p("1,000,000", small), p("Presupuesto total de tokens de entrenamiento.", small)],
                     [p("MAX_UPDATES", small), p("3,906", small), p("1,000,000 // (2 x 128): numero maximo de actualizaciones previsto.", small)],
                     [p("LEARNING_RATE", small), p("3e-4", small), p("Pico/base de learning rate; desde V4.2 se planifica con warm-up y cosine decay.", small)],
                     [p("WEIGHT_DECAY", small), p("0.1", small), p("Regularizacion aplicada con AdamW.", small)],
                     [p("EVAL_INTERVAL", small), p("10 updates", small), p("Frecuencia de evaluacion de validacion.", small)],
                     [p("VAL_BATCHES", small), p("10", small), p("Numero de batches usados por cada evaluacion de validacion.", small)]]
    story.append(line_table(common_config, [4.0*cm, 3.2*cm, 9.8*cm], True))
    section("Nota de comparabilidad")
    story.append(Table([[p("Mantener este bloque fijo hace las comparaciones posteriores mas interpretables. Aun asi, las versiones V4.1 a V4.3.3 incorporan cambios de forma acumulativa; por tanto, este control no convierte por si solo toda la trayectoria en un estudio de ablacion.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#F59E0B")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF7E6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 01", h1), p("Implementacion V4.1 - baseline de GPT", ParagraphStyle("entrysub", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: primera implementacion", small)]
    section("Objetivo")
    story.append(line_table([[p("Validar un pipeline inicial de preentrenamiento autoregresivo: arquitectura Transformer tipo GPT, tokenizador GPT-2 ya disponible y entrenamiento de una epoca. Esta ejecucion funciona como baseline para comparar iteraciones posteriores.", body)]], [17*cm]))
    section("Configuracion registrada")
    config = [[p("<b>Arquitectura</b><br/>Transformer basico tipo GPT.", small), p("<b>Tokenizacion</b><br/>Tokenizador GPT-2 ya cargado.", small)],
              [p("<b>Tamano maximo del modelo</b><br/>Aprox. 47,85 M parametros.", small), p("<b>Optimizador</b><br/>AdamW.", small)],
              [p("<b>Entrenamiento</b><br/>1 epoca sobre aprox. 1 M de tokens.", small), p("<b>Evidencia disponible</b><br/>Curva de loss de entrenamiento y validacion.", small)]]
    story.append(Table(config, colWidths=[8.5*cm, 8.5*cm], rowHeights=[1.65*cm]*3, style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Resultado observado")
    story.append(line_table([[p("El loss de validacion comunicado para esta ejecucion es <b>7,47</b>. La grafica adjunta anota una referencia final de <b>7,40711684</b>; ambas cifras quedan registradas tal como se han recibido y conviene confirmar en el log original cual corresponde al checkpoint final.", body)]], [17*cm]))
    section("Interpretacion y aprendizajes")
    story.append(line_table([[p("La curva de entrenamiento y la de validacion descienden de forma sostenida y permanecen proximas al final, por lo que esta ejecucion confirma que el pipeline aprende una senal util. El volumen de entrenamiento es muy reducido frente al tamano del modelo, asi que el resultado debe tratarse como un baseline funcional, no como una evaluacion de capacidad final. El pico inicial de train loss merece revisarse en futuras corridas si vuelve a aparecer.", body)]], [17*cm]))
    section("Siguiente paso propuesto")
    story.append(line_table([[p("Conservar configuracion, semilla, version de datos y checkpoint. Aumentar tokens de entrenamiento y comparar contra este baseline con las mismas metricas y un presupuesto de parametros controlado.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.1", h1), p("Curva de loss de entrenamiento y validacion aportada para la primera implementacion.")]
    if V41_GRAPH.exists():
        story += [Spacer(1, .2*cm), Image(str(V41_GRAPH), width=17*cm, height=9.85*cm), Spacer(1, .25*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>La validacion baja rapidamente al principio y despues se estabiliza de manera gradual. Al final, train y validacion se encuentran en el rango aproximado de 7 a 8 de loss.", body)], [p("<b>Nota de trazabilidad</b><br/>La linea discontinua del grafico esta etiquetada como 7.407116842269898. Para el informe final, confirmar si es el loss de validacion final, un promedio o el mejor valor guardado.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 02", h1), p("Implementacion V4.2 - cosine learning rate decay", ParagraphStyle("entrysub2", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: comparacion con V4.1", small)]
    section("Cambio introducido")
    story.append(line_table([[p("Se incorpora un scheduler de learning rate con <b>warm-up</b> y <b>cosine decay</b>. El learning rate crece hasta aproximadamente 3e-4 durante las primeras actualizaciones y despues disminuye suavemente hasta el final de la corrida.", body)]], [17*cm]))
    section("Condiciones de comparacion")
    compare = [[p("<b>Modelo de referencia</b><br/>GPT basico de V4.1.", small), p("<b>Presupuesto de datos</b><br/>Aprox. 1 M de tokens.", small)],
               [p("<b>Parametro cambiado</b><br/>Planificacion del learning rate: cosine decay.", small), p("<b>Lecturas comparadas</b><br/>Train loss en escala logaritmica y validation loss.", small)],
               [p("<b>Resultado comunicado</b><br/>Validation loss: 7.60 -> 7.40.", small), p("<b>Diferencia</b><br/>-0.20 puntos; mejora aproximada del 2.6% frente al baseline comunicado.", small)]]
    story.append(Table(compare, colWidths=[8.5*cm, 8.5*cm], rowHeights=[1.65*cm]*3, style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Interpretacion")
    story.append(line_table([[p("El cosine decay produce una mejora pequena pero consistente con un presupuesto de solo 1 M de tokens: la diferencia en validacion es visible, aunque no permite extraer conclusiones fuertes sobre el regimen a largo plazo. La escala logaritmica del train loss hace mas legible la separacion entre las curvas durante la convergencia; la validacion debe seguir siendo la metrica principal para escoger el cambio.", body)]], [17*cm]))
    section("Decision")
    story.append(line_table([[p("Mantener el scheduler como candidato para las siguientes corridas y repetir la comparacion con mas tokens, mismas semillas y mismo split de validacion. Reportar el valor final, el mejor valor y la varianza entre seeds.", body)]], [17*cm]))
    section("Nota de trazabilidad")
    story.append(line_table([[p("Esta comparacion usa 7.60 como baseline y 7.40 como resultado con scheduler, segun el registro aportado. La entrada V4.1 contiene otros valores historicos (7.47 y 7.407 en una grafica); antes del informe final hay que verificar que todas las cifras se refieren al mismo split, checkpoint y metodo de agregacion.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.2", h1), p("Planificacion del learning rate y comparacion de train loss con V4.1.")]
    if V42_LR_GRAPH.exists():
        story += [Spacer(1, .15*cm), Image(str(V42_LR_GRAPH), width=15.7*cm, height=8.25*cm), Spacer(1, .05*cm)]
    story.append(p("<b>Figura 1.</b> Warm-up inicial hasta aproximadamente 3e-4, seguido de un descenso suave de tipo coseno. Los valores se estiman visualmente a partir de la grafica.", small))
    if V42_TRAIN_GRAPH.exists():
        story += [Spacer(1, .15*cm), Image(str(V42_TRAIN_GRAPH), width=15.7*cm, height=8.24*cm), Spacer(1, .05*cm)]
    story.append(p("<b>Figura 2.</b> Train loss de V4.2 con scheduler frente a V4.1 en escala logaritmica. Esta escala permite comparar con mas claridad las diferencias durante la fase de convergencia.", small))
    story.append(PageBreak())

    story += [p("Evidencia - V4.2 (validacion)", h1), p("Comparacion de validation loss entre el scheduler y V4.1.")]
    if V42_VAL_GRAPH.exists():
        story += [Spacer(1, .35*cm), Image(str(V42_VAL_GRAPH), width=17*cm, height=9.48*cm), Spacer(1, .3*cm)]
    story.append(Table([[p("<b>Resultado a conservar</b><br/>El registro de la iteracion indica una reduccion de validation loss de 7.60 a 7.40 tras introducir cosine decay. La diferencia es limitada, algo esperable al entrenar solo 1 M de tokens, pero justifica realizar una corrida mas larga y controlada.", body)], [p("<b>Pregunta para la siguiente iteracion</b><br/>Con un mayor presupuesto de tokens, el scheduler mantiene o amplifica la ventaja frente al baseline? La respuesta exige comparar con las mismas semillas, datos y configuracion salvo el scheduler.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 03", h1), p("Implementacion V4.3.2 - inicializacion Xavier", ParagraphStyle("entrysub3", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: mejora acumulativa sobre V4.2", small)]
    section("Cambio introducido")
    story.append(line_table([[p("Se sustituye la inicializacion previa por <b>inicializacion Xavier</b>, manteniendo las mejoras acumuladas hasta V4.2, incluido el scheduler de cosine decay.", body)]], [17*cm]))
    section("Resultado observado")
    result = [[p("<b>Loss inicial</b><br/>La curva muestra una caida muy marcada al inicio: de aproximadamente 250 en V4.2 a aproximadamente 10 con Xavier. Es una reduccion visible de mas de un orden de magnitud (aprox. 25x).", small), p("<b>Validation loss</b><br/>El registro indica una reduccion aproximada de un punto en el conjunto de validacion/test respecto a la version anterior.", small)],
              [p("<b>Train loss</b><br/>La curva con Xavier parte mucho mas baja y se mantiene por debajo de la referencia V4.2 durante la mayor parte de la corrida.", small), p("<b>Lectura operativa</b><br/>La configuracion acumulada resulta mas estable y alcanza una mejor region de loss desde las primeras actualizaciones.", small)]]
    story.append(Table(result, colWidths=[8.5*cm, 8.5*cm], rowHeights=[2.9*cm, 2.4*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Limitacion metodologica importante")
    story.append(Table([[p("<b>Diseno incremental, no ablacion aislada.</b> Las modificaciones se han introducido acumulativamente para controlar el coste computacional y explorar rapido las opciones prometedoras. Por ello, la mejora observada en V4.3.2 no puede atribuirse de forma causal solo a Xavier: podria interactuar con el scheduler u otros cambios heredados. El resultado es valido como evidencia de una configuracion mejorada, no como una estimacion limpia del efecto individual de Xavier.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#F59E0B")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF7E6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Decision y siguiente paso")
    story.append(line_table([[p("Conservar Xavier en la configuracion operativa por su ganancia practica. Si hay presupuesto, ejecutar una ablacion con misma semilla, datos, scheduler y numero de tokens, cambiando unicamente la inicializacion. En el informe, presentar esta fase como optimizacion iterativa condicionada por recursos.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.3.2", h1), p("Comparacion de Xavier frente a V4.2 con scheduler. Naranja: Xavier; gris: V4.2.")]
    if V43_VAL_GRAPH.exists():
        story += [Spacer(1, .15*cm), Image(str(V43_VAL_GRAPH), width=15.7*cm, height=8.25*cm), Spacer(1, .05*cm)]
    story.append(p("<b>Figura 1.</b> Validation loss. Xavier inicia en una escala mucho menor y termina aproximadamente un punto por debajo de V4.2, segun el registro aportado.", small))
    if V43_TRAIN_GRAPH.exists():
        story += [Spacer(1, .15*cm), Image(str(V43_TRAIN_GRAPH), width=15.7*cm, height=5.11*cm), Spacer(1, .05*cm)]
    story.append(p("<b>Figura 2.</b> Train loss. La diferencia inicial es sustancial; esta comparacion debe interpretarse como el rendimiento de una configuracion acumulada, no como una ablacion de Xavier aislada.", small))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 04", h1), p("Implementacion V4.3.3 - activacion SwiGLU", ParagraphStyle("entrysub4", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: cambio sobre la configuracion V4.3.2", small)]
    section("Cambio introducido")
    story.append(line_table([[p("Se reemplaza la activacion anterior por <b>SwiGLU</b> en el bloque MLP del Transformer. El resto de la configuracion se mantiene segun el registro. En la terminologia del informe se usara SwiGLU, no 'Swish-GELU': es una unidad lineal con compuerta basada en Swish.", body)]], [17*cm]))
    section("Motivacion tecnica")
    story.append(line_table([[p("SwiGLU introduce una compuerta multiplicativa dentro del MLP, lo que aumenta la flexibilidad de la transformacion no lineal. La ventaja no procede de que la funcion de activacion tenga parametros entrenables por si sola, sino de la parametrizacion y expresividad del bloque con compuerta. Es una eleccion frecuente en arquitecturas modernas de modelos de lenguaje.", body)]], [17*cm]))
    section("Resultado observado")
    result_swiglu = [[p("<b>Convergencia inicial</b><br/>Las curvas son muy proximas en las primeras actualizaciones; no aparece una ventaja dominante al inicio.", small), p("<b>Mejora a largo plazo</b><br/>La curva de validation loss de SwiGLU empieza a separarse favorablemente tras mas actualizaciones y conserva una ventaja al final de la corrida.", small)],
                     [p("<b>Metrica relevante</b><br/>El test/validation loss general mejora respecto a Xavier, aunque la figura no aporta un valor final exacto para cuantificar la diferencia.", small), p("<b>Interpretacion</b><br/>El cambio parece mas beneficioso cuando el modelo ha tenido suficiente numero de sub-batches para explotar la mayor expresividad del MLP.", small)]]
    story.append(Table(result_swiglu, colWidths=[8.5*cm, 8.5*cm], rowHeights=[2.35*cm, 2.35*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Calidad de la comparacion")
    story.append(Table([[p("La trayectoria completa V4.1 -> V4.3.3 sigue siendo incremental y condicionada por recursos. Sin embargo, esta comparacion concreta es mas cercana a una ablacion que las anteriores si se confirma que datos, seed, scheduler, arquitectura y numero de tokens permanecieron iguales, cambiando solo la activacion. Esa verificacion debe constar en el informe final.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#F59E0B")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF7E6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Decision y siguiente paso")
    story.append(line_table([[p("Mantener SwiGLU como activacion candidata para corridas mas largas. Guardar los valores exactos de loss final y mejor checkpoint, y repetir la comparacion en varias seeds si el presupuesto lo permite.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.3.3", h1), p("Validation loss: SwiGLU frente a la configuracion con Xavier. Verde: SwiGLU; naranja: Xavier.")]
    if V433_VAL_GRAPH.exists():
        story += [Spacer(1, .25*cm), Image(str(V433_VAL_GRAPH), width=16.4*cm, height=11.70*cm), Spacer(1, .18*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>Las dos curvas empiezan casi solapadas. La ventaja de SwiGLU se hace visible despues de un mayor numero de actualizaciones y crece gradualmente hasta el final, lo que apoya evaluar activaciones con presupuestos de entrenamiento suficientes.", body)], [p("<b>Limitacion</b><br/>La grafica no muestra una tabla de valores exactos ni intervalos entre seeds. La conclusion es direccional: SwiGLU mejora el validation loss observado en esta corrida; la magnitud y reproducibilidad siguen por confirmar.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 05", h1), p("Implementacion V4.4 - mayor profundidad", ParagraphStyle("entrysub5", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: descartada", small)]
    section("Hipotesis y cambio propuesto")
    story.append(line_table([[p("Se prueba un Transformer mas profundo para aumentar la capacidad de modelado. La hipotesis era que mas capas mejorarian el validation/test loss manteniendo los demas componentes de la configuracion operativa.", body)]], [17*cm]))
    section("Configuracion efectiva de V4.4")
    depth_config = [[p("<b>Campo</b>", small), p("<b>Valor efectivo</b>", small), p("<b>Nota</b>", small)],
                    [p("vocab_size", small), p("50,257", small), p("Sin cambio registrado.", small)],
                    [p("context_length", small), p("128", small), p("El cfg final sobrescribe el valor 256 de GPT_CONFIG_50M con MAX_LENGTH.", small)],
                    [p("emb_dim", small), p("480", small), p("Sin cambio registrado.", small)],
                    [p("n_layers", small), p("9", small), p("Antes: 7. Este es el cambio de profundidad declarado.", small)],
                    [p("n_heads", small), p("4", small), p("Antes: 8. Cambio simultaneo que impide aislar solo la profundidad.", small)],
                    [p("ff_activation", small), p("SwiGLU", small), p("Se conserva.", small)],
                    [p("ff_hidden_dim", small), p("1,376", small), p("El cfg final sobrescribe el 1,280 de la configuracion base.", small)],
                    [p("drop_rate / qkv_bias", small), p("0.0 / False", small), p("Sin cambio registrado.", small)]]
    story.append(line_table(depth_config, [4.0*cm, 3.4*cm, 9.6*cm], True))
    section("Resultado observado")
    story.append(line_table([[p("La curva de validation loss de V4.4 (naranja) se mantiene por encima de la referencia V4.3.3 con SwiGLU (verde) durante la mayor parte del entrenamiento y termina peor. Bajo este presupuesto de tokens, la mayor profundidad no mejoro el test/validation loss y se descarta como configuracion operativa.", body)]], [17*cm]))
    section("Interpretacion y limitacion")
    story.append(Table([[p("El resultado no demuestra que los modelos mas profundos sean intrinsecamente peores: con 1 M de tokens y el mismo regimen de entrenamiento, la capacidad adicional puede requerir mas optimizacion o datos. Ademas, n_layers y n_heads cambiaron a la vez (7 -> 9 y 8 -> 4), por lo que no es una ablacion limpia de profundidad.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#F59E0B")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF7E6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Decision")
    story.append(line_table([[p("Descartar V4.4 para la linea principal y conservar V4.3.3 como referencia. Si se retoma esta hipotesis, comparar 7 frente a 9 capas manteniendo fijo el numero de heads, la dimension de embedding, la anchura del MLP y el presupuesto de tokens.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.4", h1), p("Validation loss: mayor profundidad frente a V4.3.3 con SwiGLU. Naranja: V4.4; verde: V4.3.3.")]
    if V44_VAL_GRAPH.exists():
        story += [Spacer(1, .25*cm), Image(str(V44_VAL_GRAPH), width=16.2*cm, height=8.51*cm), Spacer(1, .25*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>Las curvas parten muy cerca, pero V4.4 empieza a quedarse por encima de la referencia al avanzar el entrenamiento y termina con peor validation loss. Esto sustenta la decision de descartar la variante dentro del presupuesto evaluado.", body)], [p("<b>Aprendizaje reutilizable</b><br/>Al aumentar la profundidad, controlar las demas variables es especialmente importante. Para una ablacion interpretable, no modificar a la vez el numero de heads ni otros componentes de capacidad.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 06", h1), p("Implementacion V4.5 - Looped Transformer", ParagraphStyle("entrysub6", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: prometedora, pendiente de barrido", small)]
    section("Idea y motivacion")
    story.append(line_table([[p("Se implementa un Looped Transformer: un bloque o conjunto de bloques se reutiliza varias veces sobre la representacion. Esto aumenta el numero de transformaciones efectivas y el computo por ejemplo sin introducir un bloque independiente de parametros por cada repeticion. La idea se exploró por su interes arquitectonico; referencias externas no verificadas se consideran solo inspiracion, no evidencia sobre otros modelos.", body)]], [17*cm]))
    section("Trade-off de capacidad")
    loop_tradeoff = [[p("<b>Parametros</b><br/>La reutilizacion de pesos puede aumentar la profundidad efectiva sin un crecimiento proporcional de parametros unicos.", small), p("<b>Computo y latencia</b><br/>Cada recurrencia adicional requiere mas operaciones y puede aumentar el tiempo de entrenamiento e inferencia.", small)],
                     [p("<b>Hipotesis</b><br/>Mas pasos recurrentes permiten refinar la representacion y mejorar el validation/test loss dentro del mismo presupuesto de parametros.", small), p("<b>Variable pendiente</b><br/>Numero de recurrencias: aun no se ha determinado el valor optimo.", small)]]
    story.append(Table(loop_tradeoff, colWidths=[8.5*cm, 8.5*cm], rowHeights=[2.65*cm, 2.65*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Resultado observado")
    story.append(line_table([[p("La curva de V4.5 (verde oscuro) muestra una mejora ligera de validation/test loss frente a V4.3.3 con SwiGLU (verde claro), especialmente durante la parte media del entrenamiento. Al final las curvas convergen, por lo que la ganancia observada es moderada y debe cuantificarse con valores exactos antes de afirmarla como mejora definitiva.", body)]], [17*cm]))
    section("Decision y siguiente paso")
    story.append(Table([[p("Mantener Looped Transformer como linea de exploracion. Realizar un barrido del numero de recurrencias, registrando validation loss final, mejor checkpoint, tiempo por update, memoria y coste total. Elegir el valor optimo por calidad bajo un presupuesto de computo, no solo por numero de parametros.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#2563EB")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EAF2FF")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.5", h1), p("Validation loss: Looped Transformer frente a V4.3.3 con SwiGLU. Verde oscuro: Looped; verde claro: SwiGLU.")]
    if V45_VAL_GRAPH.exists():
        story += [Spacer(1, .25*cm), Image(str(V45_VAL_GRAPH), width=16.2*cm, height=8.51*cm), Spacer(1, .25*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>La variante recurrente se sitúa ligeramente por debajo de la referencia durante una parte importante del entrenamiento. La diferencia se reduce hacia el final, por lo que la ventaja es prometedora pero todavía no concluyente sin valores exactos y repeticiones.", body)], [p("<b>Experimento recomendado</b><br/>Probar una cuadrícula pequeña de recurrencias, por ejemplo 1, 2, 3 y 4, manteniendo fijos datos, seed y presupuesto de tokens. Para cada punto, comparar calidad y coste: loss, tiempo por update, memoria y tokens por segundo.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 07", h1), p("Implementacion V4.7 - Rotary Positional Embeddings (RoPE)", ParagraphStyle("entrysub7", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: mejora clara observada", small)]
    section("Cambio introducido")
    story.append(line_table([[p("Se reemplazan los positional embeddings por <b>Rotary Positional Embeddings (RoPE)</b>. En RoPE, la informacion de posicion se aplica mediante rotaciones a las representaciones de query y key en atencion, en lugar de sumarse como un embedding posicional aprendido independiente.", body)]], [17*cm]))
    section("Motivacion tecnica")
    rope_motivation = [[p("<b>Posicion relativa</b><br/>La interaccion entre tokens puede codificar relaciones de posicion de forma natural en el mecanismo de atencion.", small), p("<b>Parametros</b><br/>No requiere una tabla aprendida de posiciones del mismo modo que los positional embeddings absolutos.", small)],
                       [p("<b>Compatibilidad</b><br/>Se integra en Q y K de cada capa de atencion, manteniendo el resto de componentes del Transformer.", small), p("<b>Contexto de esta prueba</b><br/>El maximo efectivo de contexto sigue siendo 128 tokens; no se evalua aqui extrapolacion a longitudes mayores.", small)]]
    story.append(Table(rope_motivation, colWidths=[8.5*cm, 8.5*cm], rowHeights=[2.25*cm, 2.25*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,-1), SKY), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    section("Resultado observado")
    story.append(line_table([[p("RoPE (azul) mejora de manera clara y sostenida el validation loss frente a V4.5 Looped Transformer (verde). La separacion aparece pronto y aumenta durante el entrenamiento; al final de la grafica el loss es aproximadamente 5.8 con RoPE frente a aproximadamente 6.4 con la referencia, una diferencia visual cercana a 0.6 puntos.", body)]], [17*cm]))
    section("Interpretacion y decision")
    story.append(Table([[p("La mejora es una de las mas marcadas de las iteraciones registradas. Mantener RoPE en la configuracion principal y recuperar del log los valores exactos de loss final y mejor checkpoint. La atribucion al cambio es mas fiable si se confirma que todos los demas hiperparametros y la semilla se mantuvieron fijos frente a V4.5.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#2563EB")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EAF2FF")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Siguiente experimento recomendado")
    story.append(line_table([[p("Separar dos preguntas: (1) comparacion RoPE frente a positional embeddings a contexto fijo de 128; (2) capacidad de generalizacion al aumentar la longitud de contexto. La segunda requiere evaluar longitudes mayores, no solo reutilizar esta curva de entrenamiento.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.7", h1), p("Validation loss: RoPE frente a V4.5 Looped Transformer. Azul: RoPE; verde: Looped Transformer.")]
    if V47_VAL_GRAPH.exists():
        story += [Spacer(1, .25*cm), Image(str(V47_VAL_GRAPH), width=16.2*cm, height=8.51*cm), Spacer(1, .25*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>RoPE baja mas rapidamente tras la fase inicial y mantiene una ventaja creciente. La diferencia final visual es grande para las escalas observadas y justifica priorizar esta configuracion en el desarrollo posterior.", body)], [p("<b>Precaucion de reporte</b><br/>Los valores 5.8 y 6.4 son estimaciones visuales de la grafica. Para el informe tecnico, sustituirlos por las metricas exportadas del logger y acompañarlos de la configuracion y seed exactas.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Entrada de bitacora 08", h1), p("Implementacion V4.8 - vocabulario reducido a 16,384 IDs", ParagraphStyle("entrysub8", parent=body, fontSize=11, textColor=BLUE)), p("Fecha: no registrada    Autor/a: ____________________    Estado: mejora observada con caveat de comparabilidad", small)]
    section("Problema detectado")
    story.append(line_table([[p("La configuracion anterior usaba el vocabulario de GPT-2 de 50,257 IDs. El coste de parametros no pertenece al tokenizador como objeto, sino principalmente a las matrices del modelo dependientes del vocabulario: embedding de entrada y, segun se comparta o no peso, proyeccion de salida/softmax. Con emb_dim = 480, una sola matriz de embedding de GPT-2 ya supone aproximadamente 24.1 M de parametros.", body)]], [17*cm]))
    section("Cambio y presupuesto de parametros")
    vocab_table = [[p("<b>Aspecto</b>", small), p("<b>GPT-2</b>", small), p("<b>V4.8: tokenizer pequeno</b>", small)],
                   [p("Tamano del vocabulario", small), p("50,257 IDs", small), p("16,384 IDs", small)],
                   [p("Matriz de embedding (480 dim)", small), p("Aprox. 24.1 M parametros", small), p("Aprox. 7.9 M parametros", small)],
                   [p("Parametros totales del modelo", small), p("Aprox. 35 M", small), p("Aprox. 18 M", small)],
                   [p("Capacidad no ligada al vocabulario", small), p("Aprox. 9-10 M para MLP y atencion, segun el desglose registrado", small), p("Mas presupuesto relativo para los bloques internos o un modelo total mas pequeno", small)]]
    story.append(line_table(vocab_table, [4.3*cm, 5.6*cm, 7.1*cm], True))
    section("Resultado observado")
    story.append(line_table([[p("Pese a reducir el numero total de parametros, V4.8 con vocabulario de 16,384 IDs (morado) mejora claramente el validation/test loss respecto a V4.7 con el vocabulario GPT-2 (azul). La curva se separa pronto y termina alrededor de 4.8 frente a alrededor de 6.1 en la grafica, una mejora visual muy grande dentro de esta medicion.", body)]], [17*cm]))
    section("Interpretacion y limitacion esencial")
    story.append(Table([[p("<b>Comparabilidad de la metrica.</b> Al cambiar el tokenizador, una misma secuencia de texto se divide en un numero y unos tipos de tokens diferentes. Por eso, el cross-entropy/loss por token no es estrictamente comparable entre vocabularios. La mejora observada es prometedora, pero el informe debe complementarla con una metrica normalizada sobre el texto original, por ejemplo bits por byte (BPB), y con evaluaciones de generacion o tareas comunes.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .55, colors.HexColor("#F59E0B")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFF7E6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
    section("Decision y siguiente paso")
    story.append(line_table([[p("Mantener el vocabulario pequeno como candidato principal por eficiencia de parametros y mejora observada. Verificar que la tokenizacion se entreno y aplico sin fuga de datos, calcular BPB en el mismo texto de validacion y registrar tokens por caracter, coste y throughput junto con el loss.", body)]], [17*cm]))
    story.append(PageBreak())

    story += [p("Evidencia - V4.8", h1), p("Validation loss: vocabulario de 16,384 IDs frente al vocabulario GPT-2. Morado: V4.8; azul: V4.7 con RoPE.")]
    if V48_VAL_GRAPH.exists():
        story += [Spacer(1, .25*cm), Image(str(V48_VAL_GRAPH), width=16.2*cm, height=8.51*cm), Spacer(1, .25*cm)]
    story.append(Table([[p("<b>Lectura de la figura</b><br/>La curva con vocabulario reducido cae antes y continua por debajo de V4.7 durante toda la corrida. Es la mayor separacion observada en las graficas hasta este punto, aun cuando el modelo total tiene menos parametros.", body)], [p("<b>Como reportarlo correctamente</b><br/>Reportar esta figura como evidencia de mejor loss bajo cada tokenizacion, no como una comparacion definitiva de calidad por texto. Añadir BPB o una evaluacion comun antes de concluir que el vocabulario reducido es mejor en sentido absoluto.", body)]], colWidths=[17*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("BACKGROUND", (0,0), (-1,0), MINT), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    for idx in range(9, 10):
        story += [p(f"Entrada de bitacora {idx:02d}", h1), p("Fecha: ____________________    Autor/a: ____________________    Duracion: ____________________", small)]
        section("Objetivo de esta sesion", "Que querias averiguar, construir o corregir?")
        story.append(line_table(blank_rows(3), [17*cm]))
        section("Cambios realizados", "Codigo, datos, configuracion, dependencias o infraestructura.")
        story.append(line_table(blank_rows(4), [17*cm]))
        section("Evidencia y resultado", "Incluye metricas, tabla o enlace a grafica. Separa el hecho de la interpretacion.")
        story.append(line_table(blank_rows(4), [17*cm]))
        section("Decision y siguiente paso", "Que mantienes, cambias o descartas? Que haras a continuacion?")
        story.append(line_table(blank_rows(3), [17*cm]))
        story.append(PageBreak())

    story += [p("Registro de experimento", h1), p("Usa una hoja por ejecucion o por conjunto de ejecuciones que respondan a la misma pregunta.")]
    experiment = [[p("<b>ID del experimento</b><br/>________________________", body), p("<b>Fecha / commit</b><br/>________________________", body)],
                  [p("<b>Pregunta / hipotesis</b><br/><br/>", body), p("<b>Baseline con el que comparo</b><br/><br/>", body)],
                  [p("<b>Datos</b><br/>Version, particion, preprocesado.<br/><br/>", body), p("<b>Modelo</b><br/>Arquitectura, parametros, inicializacion.<br/><br/>", body)],
                  [p("<b>Configuracion</b><br/>LR, batch size, tokens, seed, scheduler.<br/><br/>", body), p("<b>Recursos</b><br/>GPU/CPU, tiempo, memoria, coste.<br/><br/>", body)],
                  [p("<b>Resultado cuantitativo</b><br/><br/><br/>", body), p("<b>Resultado cualitativo / muestras</b><br/><br/><br/>", body)],
                  [p("<b>Interpretacion y confianza</b><br/><br/>", body), p("<b>Conclusion / siguiente accion</b><br/><br/>", body)]]
    story.append(Table(experiment, colWidths=[8.5*cm, 8.5*cm], rowHeights=[1.6*cm, 2.2*cm, 2.6*cm, 2.6*cm, 3*cm, 2.5*cm], style=[("GRID", (0,0), (-1,-1), .45, LINE), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7)]))
    story.append(PageBreak())

    story += [p("Registro de decisiones", h1), p("Las decisiones pequeñas tambien cuentan: cambiar un tokenizer, descartar una metrica o congelar una version de datos.")]
    decisions = [[p("<b>Fecha</b>", small), p("<b>Decision</b>", small), p("<b>Alternativas y evidencia</b>", small), p("<b>Impacto / seguimiento</b>", small)]]
    for _ in range(6): decisions.append([" ", " ", " ", " "])
    story.append(line_table(decisions, [2.0*cm, 4.1*cm, 6.2*cm, 4.7*cm], True, [0.8*cm] + [1.35*cm]*6))
    story.append(Spacer(1, .5*cm))
    story += [p("Incidencias y aprendizajes", h2), p("Problema: ____________________________________________________________________________________", body), p("Causa probable: _______________________________________________________________________________", body), p("Como se resolvio / que se aprendio: ___________________________________________________________", body), p("Como prevenirlo: ______________________________________________________________________________", body), PageBreak()]

    story += [p("Del cuaderno al informe tecnico", h1), p("Al cerrar el proyecto, no redactes desde la memoria. Agrupa las entradas por estas secciones y enlaza cada afirmacion con evidencia.")]
    mapping = [[p("<b>Seccion del informe</b>", small), p("<b>Material del cuaderno que alimenta esa seccion</b>", small), p("<b>Comprobacion final</b>", small)],
               [p("Resumen ejecutivo", small), p("Objetivo, resultado principal y conclusion de las entradas.", small), p("Una frase con impacto y una metrica.", small)],
               [p("Problema y alcance", small), p("Ficha del proyecto, riesgos y exclusiones.", small), p("Evita prometer lo que no se evaluo.", small)],
               [p("Metodologia", small), p("Registros de experimento, datos, modelo y configuracion.", small), p("Permite reproducir el resultado.", small)],
               [p("Resultados", small), p("Metricas, graficas, muestras y comparacion con baseline.", small), p("Incluye resultados negativos relevantes.", small)],
               [p("Discusion", small), p("Interpretaciones, incidencias y limites anotados.", small), p("Distingue evidencia de opinion.", small)],
               [p("Conclusiones y futuro", small), p("Decisiones finales y siguientes pasos pendientes.", small), p("Lista acciones concretas y priorizadas.", small)]]
    story.append(line_table(mapping, [3.6*cm, 8.1*cm, 5.3*cm], True))
    section("Lista de cierre", "Antes de escribir: congela la version de codigo y datos, exporta las graficas finales, conserva las semillas/configuraciones y selecciona los 3-5 experimentos que mejor responden a la pregunta original.")

    story.append(PageBreak())
    story += [p("Codigo relevante - Bitacoras 01 y 02", h1), p("Fragmentos seleccionados del codigo de implementacion. Se incluyen solo las lineas que materializan cada cambio; los notebooks y modulos indicados conservan el contexto completo.")]
    code_card("Bitacora 01 - configuracion y baseline", "notebooks/Test_pretrain-v4-wandb.ipynb", """
MAX_LENGTH = 128
BATCH_SIZE = 2
MAX_TOKENS = 1_000_000
MAX_UPDATES = MAX_TOKENS // (BATCH_SIZE * MAX_LENGTH)
LEARNING_RATE = 3e-4

cfg = {**GPT_CONFIG_50M, "context_length": MAX_LENGTH}
model = GPTModel(cfg)
model.out_head.weight = model.tok_emb.weight
""")
    code_card("Bitacora 02 - warm-up y cosine decay", "src/llm_mini_lab/training/core.py", """
def cosine_lr_multiplier(step, warmup_steps, schedule_steps,
                         min_lr_ratio=0.1):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (schedule_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine

WARMUP_STEPS = int(0.05 * MAX_UPDATES)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: cosine_lr_multiplier(
        step, WARMUP_STEPS, MAX_UPDATES, 0.1))
""")

    story.append(PageBreak())
    story += [p("Codigo relevante - Bitacoras 03 y 04", h1)]
    code_card("Bitacora 03 - inicializacion Xavier", "src/llm_mini_lab/training/core.py", """
def init_xavier(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(module.weight)

model = GPTModel(cfg)
model.apply(init_xavier)
model.out_head.weight = model.tok_emb.weight
""")
    code_card("Bitacora 04 - MLP con SwiGLU", "src/llm_mini_lab/models/layers.py", """
class SwiGLU(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        self.gate_proj = nn.Linear(emb_dim, hidden_dim)
        self.value_proj = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.out_proj(gate * value)

hidden_dim = cfg.get("ff_hidden_dim", int(8 * emb_dim / 3))
self.layers = SwiGLU(emb_dim=emb_dim, hidden_dim=hidden_dim)
""")

    story.append(PageBreak())
    story += [p("Codigo relevante - Bitacoras 05 y 06", h1)]
    code_card("Bitacora 05 - variante de mayor profundidad", "notebooks/Test_pretrain-v4.4_more_depht.ipynb", """
GPT_CONFIG_50M = {
    "vocab_size": 50257, "context_length": 256, "emb_dim": 480,
    "n_heads": 4,       # antes: 8
    "n_layers": 9,      # antes: 7
    "drop_rate": 0.0, "qkv_bias": False,
    "ff_activation": "swiglu", "ff_hidden_dim": 1280,
}
cfg = {**GPT_CONFIG_50M, "context_length": MAX_LENGTH,
       "ff_activation": "swiglu", "ff_hidden_dim": 1376}
""")
    code_card("Bitacora 06 - bloques Transformer recurrentes", "src/llm_mini_lab/models/gpt.py", """
self.trf_blocks = nn.ModuleList([
    TransformerBlock(cfg) for _ in range(cfg["n_unique_layers"])
])
self.num_loops = cfg["num_loops"]

loops = self.num_loops if num_loops is None else num_loops
for _ in range(loops):
    for block in self.trf_blocks:
        x = block(x)
""")

    story.append(PageBreak())
    story += [p("Codigo de bitacoras 07 y 08", h1)]
    code_card("Bitacora 07 - Rotary Positional Embeddings", "src/llm_mini_lab/models/layers.py", """
def _apply_rope(self, x):
    cos = self.rope_cos[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = self.rope_sin[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin,
                        even * sin + odd * cos), dim=-1).flatten(-2)

if self.use_rope:
    keys = self._apply_rope(keys)
    queries = self._apply_rope(queries)
""")
    code_card("Bitacora 08 - tokenizer de 16,384 IDs y RoPE", "notebooks/v4.8_low_size_token.ipynb", """
TOKENIZER_NAME = "sp16384"
TOKENIZER_MODEL = PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model"
tokenizer = load_tokenizer(TOKENIZER_NAME, TOKENIZER_MODEL)
VOCAB_SIZE = tokenizer_vocab_size(tokenizer)

cfg = {**LOOPED_GPT_CONFIG, "context_length": MAX_LENGTH,
       "vocab_size": VOCAB_SIZE, "positional_encoding": "rope",
       "tokenizer_name": TOKENIZER_NAME}
model = LoopedGPTModel(cfg).to(device)
model.out_head.weight = model.tok_emb.weight
""")

    doc.build(story)


if __name__ == "__main__":
    build()
