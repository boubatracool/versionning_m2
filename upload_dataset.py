import os
from flask import flash, render_template, request, redirect

def allowed_file(filename, ALLOWED_EXTENSIONS):
    """ cette fonction prend en entrée le nom du fichier(avec son extension)
        et les extensions autorisés
        puis check si son extension est autorisée retourne TRUE ou FALSE si l'inverse.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file(UPLOAD_FOLDER,ALLOWED_EXTENSIONS):
    """ cette fonction prend en entrée le repertoire de destination du fichier a upload
        et les extensions autorisés
        puis effectue l'upload et envoie un message de notification.
    """
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné')
        return render_template('hello.html')

    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return render_template('hello.html')

    if file and allowed_file(file.filename,ALLOWED_EXTENSIONS):
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(filepath)
        file.save(filepath)
        flash(f'Fichier {filename} sauvegardé avec succès!')
        return render_template('hello.html')

    flash('Type de fichier non autorisé')
    return render_template('hello.html')

