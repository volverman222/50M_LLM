from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

OUT = Path('output/pdf/modelos_pequenos_factores_rendimiento.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor('#152238')
BLUE = colors.HexColor('#166D9B')
TEAL = colors.HexColor('#149C9B')
LIGHT = colors.HexColor('#EEF4F7')
MID = colors.HexColor('#D8E5EA')
INK = colors.HexColor('#202B36')
MUTED = colors.HexColor('#52626E')
ORANGE = colors.HexColor('#E98438')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=27, leading=32, textColor=NAVY, spaceAfter=12))
styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=13, leading=18, textColor=MUTED, spaceAfter=18))
styles.add(ParagraphStyle(name='H1Custom', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=NAVY, spaceBefore=2, spaceAfter=10))
styles.add(ParagraphStyle(name='H2Custom', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.5, leading=14, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontName='Helvetica', fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=10.2, leading=14, textColor=NAVY, backColor=LIGHT, borderColor=MID, borderWidth=0.5, borderPadding=9, spaceBefore=6, spaceAfter=10))
styles.add(ParagraphStyle(name='TableHead', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=7.2, leading=8.4, textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name='TableCell', parent=styles['BodyText'], fontName='Helvetica', fontSize=7.3, leading=9, textColor=INK))
styles.add(ParagraphStyle(name='TableCellCenter', parent=styles['BodyText'], fontName='Helvetica', fontSize=7.3, leading=9, textColor=INK, alignment=TA_CENTER))

def p(text, style='BodyCustom'):
    return Paragraph(text, styles[style])

def table(rows, widths, header=True, font=7.3):
    data = []
    for ri, row in enumerate(rows):
        data.append([p(str(x), 'TableHead' if header and ri == 0 else 'TableCell') for x in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    ts = [
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), .3, MID),
        ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]
    if header:
        ts += [('BACKGROUND', (0,0), (-1,0), NAVY)]
    for r in range(1 if header else 0, len(rows)):
        if r % 2 == 0: ts.append(('BACKGROUND', (0,r), (-1,r), LIGHT))
    t.setStyle(TableStyle(ts))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(MID); canvas.setLineWidth(.5)
    canvas.line(doc.leftMargin, 1.25*cm, A4[0]-doc.rightMargin, 1.25*cm)
    canvas.setFillColor(MUTED); canvas.setFont('Helvetica', 7.5)
    canvas.drawString(doc.leftMargin, .82*cm, 'Modelos pequenos de lenguaje - guia de diseño')
    canvas.drawRightString(A4[0]-doc.rightMargin, .82*cm, f'Pagina {doc.page}')
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.55*cm, leftMargin=1.55*cm, topMargin=1.55*cm, bottomMargin=1.75*cm)
story = []

# Cover
story += [Spacer(1, 2.4*cm), p('QUE HACE COMPETITIVO A UN MODELO PEQUENO', 'TitleCustom'),
          p('Guia practica para LLMs de menos de 100 millones de parametros', 'Subtitle'),
          Spacer(1, .45*cm), p('Tesis principal', 'H2Custom'),
          p('En el rango 10M-100M, el numero de parametros es solo una parte de la historia. La calidad y mezcla de datos, el tokenizer, el presupuesto de tokens, el objetivo de entrenamiento y el protocolo de evaluacion pueden cambiar el resultado mas que duplicar el tamano del modelo.', 'Callout'),
          Spacer(1, .35*cm),
          table([
              ['Decisiones que multiplican valor', 'Efecto esperado'],
              ['Datos curados y curriculum', 'Mas conocimiento util por parametro; menos capacidad desperdiciada.'],
              ['Tokenizer adaptado al dominio', 'Menos fragmentacion; secuencias mas cortas y patrones mas faciles de aprender.'],
              ['Destilacion o ajuste por tarea', 'Transfiere comportamiento de modelos grandes a una red compacta.'],
              ['Evaluacion honesta', 'Separa progreso real de contaminacion, prompts favorables o azar.'],
          ], [6.2*cm, 10.3*cm]),
          Spacer(1, 1.0*cm), p('Alcance: modelos autoregresivos base de la tabla proporcionada, mas lecciones aplicables a modelos pequenos en general.', 'Small'), PageBreak()]

# Context and reading table
story += [p('1. Como leer los resultados', 'H1Custom'),
          p('Las cinco pruebas miden capacidades distintas. ARC-Easy y ARC-Challenge son preguntas de ciencia escolar; HellaSwag mide continuacion de sentido comun; PIQA, razonamiento fisico cotidiano; y ArithMark, aritmetica. En ARC y HellaSwag hay cuatro opciones: 25% es aproximadamente azar. En PIQA hay dos: 50% es aproximadamente azar.', 'BodyCustom'),
          p('Por eso un 31% en HellaSwag indica una mejora, pero todavia pequena; y un 23%-25% en ARC-Challenge se debe interpretar con mucha cautela. El rendimiento util depende de la tarea final, no de un promedio ciego.', 'Callout'),
          table([
              ['Senal en la tabla', 'Lectura correcta', 'Implicacion'],
              ['Supra-1.5-50M lidera ARC-Easy (48.40)', 'Fuerte para conocimiento escolar basico a 51.8M.', 'Candidato generalista eficiente.'],
              ['Veyra2 lidera PIQA (62.13) y HellaSwag (31.28)', 'Mejor sentido comun / fisica cotidiana entre los 50M.', 'Buen punto de partida para asistentes de texto cortos.'],
              ['Gros-Michel-90M v2 lidera ArithMark (35.90)', 'Mejor perfil numerico de la lista.', 'Elegir si las respuestas contienen calculo.'],
              ['Pythia-70M queda bajo Pythia-31M', 'Mas parametros sin receta mejor no garantizan mejora.', 'Los datos y el entrenamiento dominan en esta escala.'],
          ], [4.1*cm, 6.4*cm, 6.0*cm]),
          Spacer(1, .25*cm),
          p('<b>Promedio simple orientativo:</b> Supra ~39.6, Gros-Michel v2 ~39.2, BananaMind ~38.4, Veyra2 ~38.2 y Nexus ~37.1. No es una metrica oficial: mezcla tareas, escalas y posibles conjuntos de entrenamiento diferentes.', 'Small'), PageBreak()]

# components
story += [p('2. Que aporta cada componente', 'H1Custom'),
          table([
              ['Componente', 'Que cambia dentro del modelo', 'Por que pesa mas en 10M-100M'],
              ['Arquitectura', 'Profundidad, ancho, atencion, normalizacion, MLP y longitud de contexto.', 'Define donde cabe la capacidad, pero no crea conocimiento por si sola. Debe ser facil de optimizar.'],
              ['Tokenizer', 'Convierte texto a tokens. Decide vocabulario, fragmentacion y manejo de numeros/codigo.', 'Cada token ahorrado equivale a mas contexto y mas ejemplos vistos con el mismo compute.'],
              ['Datos de preentrenamiento', 'Aportan lenguaje, hechos, estilo y patrones de razonamiento.', 'Un modelo pequeno no puede memorizar ruido: necesita alta densidad de senal.'],
              ['Mezcla y curriculum', 'Ordena y pondera documentos: texto general, educativo, codigo, QA, sintetico.', 'Permite especializar sin perder fluidez; evita que una fuente masiva domine todo.'],
              ['Objetivo de entrenamiento', 'Prediccion del siguiente token, mascara, denoising o tareas auxiliares.', 'Induce las habilidades que el modelo practicara.'],
              ['Distilacion / SFT', 'Imita distribuciones o respuestas de un profesor grande.', 'Es la via mas directa para adquirir conducta util sin aumentar parametros.'],
              ['Inferencia', 'Prompt, formato de opciones, temperatura, longitud de salida.', 'Puede desbloquear o esconder capacidad ya aprendida, pero no sustituye entrenamiento.'],
          ], [2.5*cm, 6.1*cm, 7.9*cm]),
          p('Regla operativa: antes de aumentar de 50M a 90M, revisar cobertura de datos, ratio tokens/parametro, tokenizer y errores de validacion. Es habitual obtener mas ganancia con una segunda pasada de datos curados que con una red mayor entrenada de forma identica.', 'Callout'), PageBreak()]

# deep dives
story += [p('3. Datos y tokenizer: donde nacen las diferencias', 'H1Custom'),
          p('Datos curados. Un corpus educativo, de preguntas-respuestas, ciencia y explicaciones fortalece ARC; historias y situaciones humanas ayudan HellaSwag; ejemplos de acciones fisicas ayudan PIQA. El riesgo es que entrenar sobre datos demasiado cercanos a un benchmark mide recuperacion o contaminacion, no generalizacion.', 'BodyCustom'),
          p('Mezcla. La proporcion importa. Mucho texto web puede mejorar fluidez y empeorar precision; demasiados datos sinteticos pueden reforzar formatos repetitivos. Una mezcla saludable combina texto de alta calidad, material pedagogico, dominio objetivo y una fraccion sintetica validada.', 'BodyCustom'),
          p('Tokenizer. Un tokenizer de 32k tokens puede ser mas eficiente que uno generico si reduce fragmentacion en el dominio. Para matematicas, mantener digitos y operadores en unidades coherentes facilita aprender estructura posicional. Nexus-Erebus declara precisamente un tokenizer consciente de digitos, una explicacion plausible para su perfil aritmetico, aunque no sustituye una evaluacion externa.', 'BodyCustom'),
          table([
              ['Decisión', 'Aporta', 'Riesgo / control'],
              ['Filtrar duplicados, basura y texto roto', 'Menos gradientes inutiles; mejora calidad por token.', 'No filtrar material raro pero valioso; auditar muestras.'],
              ['Sobremuestrear conocimiento objetivo', 'Mueve el modelo hacia la tarea del producto.', 'Sobreajuste tematico; conservar un nucleo generalista.'],
              ['Datos sinteticos con verificador', 'Escala ejemplos de calculo, formatos y explicaciones.', 'Errores del generador; usar reglas, ejecucion o respuestas conocidas.'],
              ['Tokenizer de dominio', 'Menos tokens por ejemplo y patrones mas consistentes.', 'Compatibilidad menor con modelos y corpus externos.'],
          ], [3.8*cm, 6.0*cm, 6.7*cm]), PageBreak()]

# model mapping
story += [p('4. Que parece aportar a cada familia de tu tabla', 'H1Custom'),
          p('No todas las tarjetas publican el mismo detalle de entrenamiento; por tanto esta es una interpretacion de los patrones observados, no una atribucion causal demostrada. La forma correcta de confirmarla es una ablation controlada.', 'Small'),
          table([
              ['Modelo / grupo', 'Patron observado', 'Hipotesis de valor', 'Uso recomendado'],
              ['Supra-1.5-50M Base', 'Mejor ARC-Easy y mejor promedio simple.', 'Receta de datos/entrenamiento generalista especialmente eficiente.', 'Baseline principal para 50M.'],
              ['Veyra2-Apricot-50M', 'Mejor HellaSwag y PIQA.', 'Datos orientados a sentido comun y buena cobertura cotidiana.', 'Fluidez, clasificacion y QA cotidiana.'],
              ['Gros-Michel-90M v2', 'Mejor ArithMark; v2 mejora a la version previa.', 'El cambio de receta importa mas que el salto de parametros; posible mejor mezcla numerica.', 'Candidata si se permite 95M.'],
              ['Nexus-Erebus-50M', 'Competitivo en ARC-Easy y aritmetica declarada.', 'Tokenizer de digitos y curriculum sintetico de enteros.', 'Experimentos matematicos; validar contaminacion.'],
              ['Pythia 14M-70M', 'Resultados desiguales al escalar.', 'Referencia de entrenamiento, no techo de capacidad actual.', 'Control reproducible de ablations.'],
              ['SLM-10M / GPT-S 5M', 'PIQA muy por encima de azar con capacidades limitadas.', 'Incluso micro-modelos absorben regularidades simples.', 'Demos on-device y experimentos educativos.'],
          ], [3.0*cm, 3.8*cm, 5.4*cm, 4.3*cm]),
          p('La comparacion Supra vs. modelos de 90M es la leccion mas accionable: si dos modelos comparten la misma evaluacion, ganar ARC-Easy con 51.8M frente a opciones de 75M-95M sugiere que el presupuesto debe ir primero a datos y a iteraciones de receta.', 'Callout'), PageBreak()]

# methodology
story += [p('5. Como demostrar que una mejora es real', 'H1Custom'),
          p('Un leaderboard es un filtro inicial, no la decision final. Re-evaluar los checkpoints candidatos con el mismo codigo, commit de harness, prompt, numero de ejemplos (0-shot / few-shot), normalizacion de probabilidad y conjuntos completos. Registrar semilla y version de cada artefacto.', 'BodyCustom'),
          table([
              ['Prueba', 'Que evita', 'Decision que habilita'],
              ['Conjunto interno en espanol y fuera de distribucion', 'Elegir por benchmarks ingleses que no representan el producto.', 'Seleccion de modelo para el caso de uso real.'],
              ['Ablation: mismo modelo, un factor a la vez', 'Atribuir una ganancia al parametro equivocado.', 'Saber si gano el tokenizer, datos o entrenamiento.'],
              ['Contaminacion: deduplicar contra test y buscar n-gramas', 'Memorizacion de preguntas de benchmark.', 'Credibilidad del resultado.'],
              ['Curvas de perdida y de benchmark por tokens', 'Parar demasiado pronto o sobreentrenar una mezcla.', 'Asignacion eficiente de compute.'],
              ['Latencia, RAM y tokens/s con el hardware objetivo', 'Optimizar solo accuracy.', 'Viabilidad de despliegue.'],
          ], [4.3*cm, 5.5*cm, 6.7*cm]),
          p('Protocolo minimo para el hackathon: (1) elegir Supra, Veyra2 y Gros-Michel v2; (2) correr los cinco benchmarks con un harness fijado; (3) crear 200-500 items propios en espanol, separados de entrenamiento; (4) comparar exactitud, latencia y memoria; (5) inspeccionar manualmente 30 fallos por modelo.', 'Callout'),
          p('Recomendacion final: para un limite estricto de 50M, empezar con Supra-1.5-50M y Veyra2-Apricot-50M. Si calculo es central o se admite 95M, incluir Gros-Michel-90M v2. Usar Pythia-31M como control historico, no como candidato ganador.', 'BodyCustom'), PageBreak()]

# sources
story += [p('6. Fuentes y notas de reproducibilidad', 'H1Custom'),
          p('La tabla de partida fue facilitada por el usuario. Los resultados deben interpretarse solo si los modelos se evaluaron con el mismo protocolo. Las fuentes siguientes documentan conceptos y ejemplos mencionados en el informe.', 'BodyCustom'),
          p('<b>[1]</b> Microsoft, MiniLM README. Resultados de destilacion y comparacion de parametros. https://github.com/microsoft/unilm/blob/master/minilm/README.md', 'Small'),
          p('<b>[2]</b> Wang et al. (2020), MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression. https://arxiv.org/abs/2002.10957', 'Small'),
          p('<b>[3]</b> Chung et al. (2022), Scaling Instruction-Finetuned Language Models / FLAN Collection. https://arxiv.org/abs/2210.11416', 'Small'),
          p('<b>[4]</b> AxiomicLabs, Open SLM Leaderboard y configuracion de entradas. https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard', 'Small'),
          p('<b>[5]</b> MaliosDark, Nexus-Erebus-50M model card: arquitectura, tokenizer y protocolo declarado. https://huggingface.co/MaliosDark/Nexus-Erebus-50M', 'Small'),
          p('<b>[6]</b> AxiomicLabs, ArithMark 3.0 dataset card. https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0', 'Small'),
          Spacer(1, .4*cm), p('Nota de cautela', 'H2Custom'),
          p('Una puntuacion publicada por la propia tarjeta de un modelo es evidencia util, pero no independiente. Prioriza benchmarks reproducibles, conjuntos de prueba no vistos y comparaciones con versiones fijadas. Los resultados de ArithMark-2 y ArithMark-3 no son intercambiables; no deben compararse como si fuesen la misma metrica.', 'Callout')]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT.resolve())
