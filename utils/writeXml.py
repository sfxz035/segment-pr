from xml.dom.minidom import Document

patient = 'None'

def writeInfoToXml(writepath,collection):
    file = collection.get('file')
    nub = collection.get('nub')
    shape = collection.get('shape')
    wid,heg,dep = str(shape[0]),str(shape[1]),str(shape[2])
    boxes = collection.get('boxes')
    doc = Document()
    annList = doc.createElement('annotation')
    doc.appendChild(annList)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode(patient))

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(file))

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('Unknown'))
    source.appendChild(database)

    size = doc.createElement('size')
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')
    width.appendChild(doc.createTextNode(wid))
    height.appendChild(doc.createTextNode(heg))
    depth.appendChild(doc.createTextNode(dep))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))

    annList.appendChild(folder)
    annList.appendChild(filename)
    annList.appendChild(source)
    annList.appendChild(size)
    annList.appendChild(segmented)

    for i in range(nub):
        box = boxes[i]
        xminget = str(box[0])
        yminget = str(box[1])
        xmaxget = str(box[2])
        ymaxget = str(box[3])
        object = doc.createElement('object')

        name = doc.createElement('name')
        pose = doc.createElement('pose')
        truncated = doc.createElement('truncated')
        diffcult = doc.createElement('diffcult')
        bndbox = doc.createElement('bndbox')

        name.appendChild(doc.createTextNode('fiducial'))
        pose.appendChild(doc.createTextNode('pose'))
        truncated.appendChild(doc.createTextNode('0'))
        diffcult.appendChild(doc.createTextNode('0'))

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(xminget))
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(yminget))
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(xmaxget))
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(ymaxget))

        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(diffcult)
        object.appendChild(bndbox)
        annList.appendChild(object)


    with open(writepath, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

